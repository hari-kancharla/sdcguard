"""Canonicalize only existing ordinary CPU paths and damage inert receipts."""
from copy import deepcopy
import json
import socket
import subprocess

import pytest

from sdcguard import evidence_components as adapter
from sdcguard.lifetime import component_conformance as component
from sdcguard.lifetime.generation_obligations import evaluate_generation_obligations
from sdcguard.lifetime.generation_observations import sglang_component_observations, vllm_probe_observations


@pytest.fixture(scope="module")
def raw():
    return adapter.collect()


def rederive(result):
    """Update only derived claims after an inert evidence deletion."""
    for engine,item in result["engines"].items():
        receipt=item["receipt"]
        translate=vllm_probe_observations if engine=="vllm" else sglang_component_observations
        item["assessment"]=evaluate_generation_obligations(translate(receipt["observations"],receipt["metadata"])).as_dict()
        item["observation_count"]=len(receipt["observations"])
    result["status"]="satisfied" if all(x["assessment"]["status"]=="satisfied" for x in result["engines"].values()) else "unknown"
    return result


def test_both_ordinary_engines_feed_one_named_rule_without_physical_safety_claim(raw):
    findings=adapter.normalize(raw)
    assert len(findings)==2
    assert {f["rule_id"] for f in findings}=={adapter.RULE_ID}
    assert {f["semantic_revision"] for f in findings}=={1}
    assert {f["evidence_kind"] for f in findings}=={"source_fragment_cpu"}
    assert {f["scope"] for f in findings}=={"source_fragment_cpu"}
    assert {f["verdict"] for f in findings}=={"satisfied"}
    assert {f["resource_kind"] for f in findings}=={"destination_kv_allocation","protocol_held_auxiliary_metadata"}
    for finding in findings:
        assert set(finding)=={"rule_id","semantic_revision","subject","scenario","evidence_kind","scope","resource_kind","verdict","facts","missing","evidence"}
        assert finding["facts"]["physical_write_safety"]=="unknown"
        assert finding["facts"]["engine_version"]=="unknown"
        assert finding["facts"]["obligations"][0]["resource"]
        assert finding["facts"]["obligations"][0]["operation"]
        assert finding["facts"]["source"]["commit"]
        assert finding["facts"]["source"]["sha256"]
        assert finding["facts"]["generation_basis"]==finding["facts"]["obligations"][0]["generation_basis"]
        assert not finding["missing"]


def test_json_roundtrip_preserves_finding_semantics_and_opaque_receipts(raw):
    assert adapter.normalize(raw)==adapter.normalize(json.loads(json.dumps(raw)))
    before=deepcopy(raw)
    adapter.normalize(raw)
    assert before==raw


def test_findings_fit_the_shared_catalog_without_redefining_its_scope(raw):
    from sdcguard.evidence_catalog import validate_finding
    for finding in adapter.normalize(raw):
        assert validate_finding(finding,"source_fragment_cpu")==[]


def test_normalization_never_collects_or_executes_a_component(raw,monkeypatch):
    def refuse(*args,**kwargs):raise AssertionError("normalization attempted execution")
    monkeypatch.setattr(component,"run_component_conformance",refuse)
    monkeypatch.setattr(component,"_methods",refuse)
    assert all(f["verdict"]=="satisfied" for f in adapter.normalize(raw))


def test_fresh_collection_is_a_separate_witness_not_a_byte_replay(raw):
    fresh=adapter.collect()
    assert fresh!=raw
    a,b=adapter.normalize(raw),adapter.normalize(fresh)
    assert [(x["subject"],x["scenario"],x["verdict"]) for x in a]==[(x["subject"],x["scenario"],x["verdict"]) for x in b]
    assert a[0]["evidence"]["receipt"]!=b[0]["evidence"]["receipt"]


@pytest.mark.parametrize("selection",[None,True,[],{},"other","all --source anything"])
def test_collect_refuses_unmapped_selection_without_execution(selection,monkeypatch):
    monkeypatch.setattr(component,"run_component_conformance",lambda *a:pytest.fail("unexpected runner call"))
    with pytest.raises(ValueError):adapter.collect(selection)


@pytest.mark.parametrize("engine",["vllm","sglang"])
def test_standalone_collection_cannot_masquerade_as_complete_portable_profile(engine):
    result=adapter.collect(engine)
    assert set(result["engines"])=={engine}
    with pytest.raises(ValueError):adapter.normalize(result)


@pytest.mark.parametrize("field",["schema","scope","physical_write_safety","evidence_origin","engines"])
def test_required_root_contract_is_exact(raw,field):
    result=deepcopy(raw);result.pop(field)
    with pytest.raises(ValueError):adapter.normalize(result)


@pytest.mark.parametrize("field,value",[("schema","other"),("scope","deployment"),
    ("physical_write_safety","satisfied"),("evidence_origin","deployed engine")])
def test_unsupported_scope_or_origin_cannot_be_upgraded(raw,field,value):
    result=deepcopy(raw);result[field]=value
    with pytest.raises(ValueError):adapter.normalize(result)


@pytest.mark.parametrize("engine",["vllm","sglang"])
@pytest.mark.parametrize("where",["item","metadata"])
@pytest.mark.parametrize("damage",["commit","missing_hash","changed_hash","extra_hash"])
def test_both_source_claims_require_exact_commit_and_full_source_inventory(raw,engine,where,damage):
    result=deepcopy(raw);item=result["engines"][engine]
    pin=item["source"] if where=="item" else item["receipt"]["metadata"]["source"]
    if damage=="commit":pin["commit"]="0"*40
    elif damage=="missing_hash":pin["sha256"].pop(next(iter(pin["sha256"])))
    elif damage=="changed_hash":pin["sha256"][next(iter(pin["sha256"]))]="0"*64
    else:pin["sha256"]["unexpected.py"]="0"*64
    with pytest.raises(ValueError):adapter.normalize(result)


@pytest.mark.parametrize("engine",["vllm","sglang"])
@pytest.mark.parametrize("field",["source","assessment","observation_count","receipt"])
def test_item_schema_cannot_drop_claims_or_raw_receipt(raw,engine,field):
    result=deepcopy(raw);result["engines"][engine].pop(field)
    with pytest.raises(ValueError):adapter.normalize(result)


def test_missing_or_added_engine_inventory_is_rejected(raw):
    result=deepcopy(raw);result["engines"].pop("sglang")
    with pytest.raises(ValueError):adapter.normalize(result)
    result=deepcopy(raw);result["engines"]["unreviewed"]=deepcopy(result["engines"]["vllm"])
    with pytest.raises(ValueError):adapter.normalize(result)


@pytest.mark.parametrize("field,value",[("status","unknown"),("reasons",["invented"]),
    ("physical_write_safety","satisfied"),("obligations",[])])
def test_claimed_assessment_is_not_trusted_when_it_disagrees_with_raw_facts(raw,field,value):
    result=deepcopy(raw);result["engines"]["vllm"]["assessment"][field]=value
    with pytest.raises(ValueError):adapter.normalize(result)


def test_claimed_aggregate_status_must_match_recomputation(raw):
    result=deepcopy(raw);result["status"]="unknown"
    with pytest.raises(ValueError):adapter.normalize(result)


@pytest.mark.parametrize("engine",["vllm","sglang"])
@pytest.mark.parametrize("damage",["no_events","coverage","clock","sequence_gap","args","pool","closed"])
def test_missing_recorded_proof_remains_unknown_not_alarm(raw,engine,damage):
    result=deepcopy(raw);receipt=result["engines"][engine]["receipt"]
    if damage=="no_events":receipt["observations"]=[]
    elif damage=="coverage":receipt["metadata"].pop("complete_from_start")
    elif damage=="clock":receipt["observations"][2].pop("clock_id")
    elif damage=="sequence_gap":receipt["observations"][2]["seq"]+=1
    elif damage=="args":
        kind="alloc" if engine=="vllm" else "drain_ack"
        next(x for x in receipt["observations"] if x["kind"]==kind)["args"]={}
    elif damage=="pool":receipt["metadata"].pop("pool" if engine=="vllm" else "metadata_pool")
    else:receipt["metadata"]["closed"]=False
    finding=next(x for x in adapter.normalize(rederive(result)) if x["subject"]=={"engine":engine})
    assert finding["verdict"]=="unknown" and finding["missing"]
    assert finding["facts"]["physical_write_safety"]=="unknown"


@pytest.mark.parametrize("engine",["vllm","sglang"])
def test_raw_mutation_cannot_keep_the_previous_positive_derived_verdict(raw,engine):
    result=deepcopy(raw);result["engines"][engine]["receipt"]["metadata"]["closed"]=False
    with pytest.raises(ValueError):adapter.normalize(result)


@pytest.mark.parametrize("damage",["args_type","kind","unknown_envelope_key","nonfinite","count_bool",
    "count_wrong","receipt_type","metadata_type","coverage_type","pid_type","unexpected_metadata"])
def test_malformed_receipts_raise_integrity_error(raw,damage):
    result=deepcopy(raw);item=result["engines"]["vllm"];receipt=item["receipt"];event=receipt["observations"][1]
    if damage=="args_type":event["args"]=[]
    elif damage=="kind":event["kind"]="unreviewed"
    elif damage=="unknown_envelope_key":event["run_code"]="raise AssertionError()"
    elif damage=="nonfinite":event["mono"]=float("nan")
    elif damage=="count_bool":item["observation_count"]=True
    elif damage=="count_wrong":item["observation_count"]+=1
    elif damage=="receipt_type":item["receipt"]=[]
    elif damage=="metadata_type":receipt["metadata"]=[]
    elif damage=="coverage_type":receipt["metadata"]["closed"]=1
    elif damage=="pid_type":event["pid"]=True
    else:receipt["metadata"]["source_code"]="raise AssertionError()"
    with pytest.raises(ValueError):adapter.normalize(result)


@pytest.mark.parametrize("field",["run_performed","requested_engine","source_path"])
def test_extra_execution_claim_or_caller_source_path_is_not_accepted(raw,field):
    result=deepcopy(raw);result[field]="untrusted"
    with pytest.raises(ValueError):adapter.normalize(result)


def test_custom_python_mapping_methods_are_not_executed_as_receipt_data(raw):
    class UntrustedDict(dict):
        def values(self):raise AssertionError("untrusted mapping executed")
    with pytest.raises(ValueError):adapter.normalize(UntrustedDict(raw))


def test_circular_or_non_json_data_is_an_integrity_error(raw):
    result=deepcopy(raw);result["engines"]["vllm"]["receipt"]["metadata"]["pool"]=result
    with pytest.raises(ValueError):adapter.normalize(result)


def test_receipt_cannot_exceed_canonical_catalog_nesting(raw):
    result=deepcopy(raw);nested={}
    for _ in range(70):nested={"nested":nested}
    result["engines"]["vllm"]["receipt"]["observations"][0]["args"]["unused"]=nested
    with pytest.raises(ValueError):adapter.normalize(result)


def test_collect_uses_existing_paths_without_network_process_or_engine_import(monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name.split('.')[0] not in {"torch","vllm","sglang","lmcache"}
        return original(name,*args,**kwargs)
    def refuse(*args,**kwargs):raise AssertionError("unexpected process or network action")
    monkeypatch.setattr(builtins,"__import__",guarded)
    monkeypatch.setattr(socket,"create_connection",refuse)
    monkeypatch.setattr(socket.socket,"connect",refuse)
    monkeypatch.setattr(socket.socket,"sendto",refuse)
    monkeypatch.setattr(subprocess,"Popen",refuse)
    findings=adapter.normalize(adapter.collect())
    assert all(x["verdict"]=="satisfied" for x in findings)
