"""Local file and host-metadata controls. Owner: Hari Krishna Kancharla.

Small file fixtures are not model weights or a GPU execution witness.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import textwrap
from types import SimpleNamespace as NS
import weakref

import pytest

from sdcguard import model_reference as M
from sdcguard import forward_report, assignment_report
from sdcguard.lifetime import forward_inputs as F, assignment_obligations as A, vllm_forward as V
from sdcguard.lifetime import mapping_runtime as R
from test_forward_inputs import ordinary_records as forward_records
from test_assignment_obligations import ordinary_records as assignment_records
from test_vllm_forward import environment
from test_mapping_runtime import header_environment
from test_assignment_init import primary


RUN, PACKET = "1" * 32, "2" * 64


class WeakFields(NS):
    pass


def local_config(runner, fields):
    config = WeakFields(**vars(runner.model_config))
    vars(config).update(vars(fields))
    runner.model_config = runner.vllm_config.model_config = config
    return config


@pytest.fixture
def local_snapshot(tmp_path, monkeypatch):
    """Ten bounded host files exercise IO independently of any model runtime."""
    actual_pin = M._pin()
    root = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / M.REVISION
    root.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    pin = dict(model=M.MODEL, revision=M.REVISION, files={})
    for index, name in enumerate(actual_pin["files"]):
        raw = ("host file fixture " + name).encode()
        blob = blobs / str(index)
        blob.write_bytes(raw)
        (root / name).symlink_to(blob)
        pin["files"][name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(M, "_pin", lambda: deepcopy(pin))
    capsule = M.capture(root, run_id=RUN, packet_sha256=PACKET)
    path = tmp_path / "model-reference.json"
    path.write_text(json.dumps(capsule, sort_keys=True))
    monkeypatch.setenv(M.ENV_FILE, str(path))
    monkeypatch.setenv(M.ENV_RUN, RUN)
    monkeypatch.setenv(M.ENV_PACKET, PACKET)
    config = WeakFields(model=str(root), revision=M.REVISION, tokenizer=str(root), tokenizer_revision=M.REVISION)
    return NS(root=root, path=path, capsule=capsule, config=config, pin=pin)


def test_installed_pin_is_exact_reviewed_resource():
    pin = M._pin()
    assert pin["model"] == M.MODEL and pin["revision"] == M.REVISION
    assert len(pin["files"]) == 10


def test_original_offline_branch_resolves_model_and_tokenizer_without_aliasing(tmp_path):
    folder = Path(__file__).parent / "fixtures/model_reference"
    raw = (folder / "arg_utils_offline.py.txt").read_bytes()
    source = json.loads((folder / "SOURCE.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == source["excerpt_sha256"]
    assert source["commit"] == "5df63a7203cdb9328d824a27b682e77c7b9928be"
    assert source["full_file_sha256"] == "c10281277fd9a90dc8b22670db4f1af53708c5caa2aefee7aa861fa579329219"
    resolved = str(tmp_path / "snapshots" / M.REVISION)
    config = NS(model=M.MODEL, revision=M.REVISION, tokenizer=M.MODEL, tokenizer_revision=M.REVISION)
    calls = []
    def resolve(model, revision):
        calls.append((model, revision))
        return resolved
    namespace = dict(self=config, huggingface_hub=NS(constants=NS(HF_HUB_OFFLINE=True)),
                     is_cloud_storage=lambda value: False, get_model_path=resolve,
                     logger=NS(info=lambda *args: None))
    exec(compile(textwrap.dedent(raw.decode()), source["source"], "exec"), namespace)
    assert config.model == config.tokenizer == resolved
    assert calls == [(M.MODEL, M.REVISION), (M.MODEL, M.REVISION)]
    assert config.revision == config.tokenizer_revision == M.REVISION


def test_local_binding_retains_observed_path_and_separate_canonical_model(local_snapshot):
    s = local_snapshot
    binding = M.ModelBinding(s.config)
    result = binding.observe(s.config)
    assert s.config.model == result["reference"]["configured_snapshot"]
    assert result["reference"]["model"] == M.MODEL
    assert result["tokenizer"] == s.config.tokenizer
    assert result["reference"]["loaded_weights"] == "not observed"
    assert result["reference"]["authenticity"] == "not established"
    result["reference"]["files"].clear()
    assert len(binding.observe(s.config)["reference"]["files"]) == 10
    with pytest.raises(FrozenInstanceError):
        binding._fields = ()


def test_hot_checks_do_not_hash_or_read_model_files(local_snapshot, monkeypatch):
    s = local_snapshot
    binding = M.ModelBinding(s.config)
    def refuse(*args, **kwargs):
        raise AssertionError("hot check read model files or rebuilt proof")
    monkeypatch.setattr(M, "_current_files", refuse)
    monkeypatch.setattr(M, "_pin", refuse)
    monkeypatch.setattr(M.os, "read", refuse)
    for _ in range(3):
        assert binding.observe(s.config)["tokenizer"] == s.config.tokenizer


@pytest.mark.parametrize("field,value", [
    ("model", M.MODEL), ("model", "/tmp/alias"), ("revision", "0" * 40),
    ("tokenizer", M.MODEL), ("tokenizer_revision", "main"),
])
def test_changed_native_reference_never_rebinds_silently(local_snapshot, field, value):
    s = local_snapshot
    binding = M.ModelBinding(s.config)
    setattr(s.config, field, value)
    with pytest.raises(M.ModelReferenceError):
        binding.observe(s.config)


@pytest.mark.parametrize("field", ["model", "revision", "tokenizer", "tokenizer_revision"])
def test_runtime_reference_fields_keep_exact_observed_types(local_snapshot, field):
    class OtherReference(str):
        pass
    s = local_snapshot
    binding = M.ModelBinding(s.config)
    setattr(s.config, field, OtherReference(getattr(s.config, field)))
    with pytest.raises(M.ModelReferenceError):
        binding.observe(s.config)


def test_replaced_config_or_capsule_is_not_the_same_binding(local_snapshot):
    s = local_snapshot
    binding = M.ModelBinding(s.config)
    with pytest.raises(M.ModelReferenceError):
        binding.observe(NS(**vars(s.config)))
    raw = s.path.read_bytes()
    replacement = s.path.with_suffix(".replacement")
    replacement.write_bytes(raw)
    os.replace(replacement, s.path)
    with pytest.raises(M.ModelReferenceError):
        binding.observe(s.config)


def test_binding_does_not_retain_native_model_config(local_snapshot):
    config = WeakFields(**vars(local_snapshot.config))
    binding = M.ModelBinding(config)
    reference = weakref.ref(config)
    del config
    gc.collect()
    assert reference() is None
    with pytest.raises(M.ModelReferenceError):
        binding.observe(WeakFields(**vars(local_snapshot.config)))


def test_nonweak_local_config_is_unsupported(local_snapshot):
    with pytest.raises(M.ModelReferenceError, match="weak-referenceable"):
        M.ModelBinding(NS(**vars(local_snapshot.config)))


@pytest.mark.parametrize("name,value", [(M.ENV_RUN, "3" * 32), (M.ENV_PACKET, "4" * 64)])
def test_another_run_or_packet_capsule_is_not_accepted(local_snapshot, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(M.ModelReferenceError):
        M.ModelBinding(local_snapshot.config)


def test_missing_capsule_and_model_alias_are_not_proof(local_snapshot, monkeypatch):
    s = local_snapshot
    monkeypatch.delenv(M.ENV_FILE)
    with pytest.raises(M.ModelReferenceError):
        M.ModelBinding(s.config)
    s.config.model = "served-model-alias"
    with pytest.raises(M.ModelReferenceError):
        M.bind(s.config)


@pytest.mark.parametrize("change", [
    lambda c: c.update(revision="0" * 40),
    lambda c: c.update(model_pin_sha256="0" * 64),
    lambda c: c.update(configured_snapshot="/tmp/other/" + M.REVISION),
    lambda c: c["files"].pop("README.md"),
    lambda c: c["files"]["README.md"].update(sha256="0" * 64),
    lambda c: c["files"]["README.md"].update(bytes=True),
    lambda c: c.update(authenticity="verified"),
    lambda c: c.update(loaded_weights="verified"),
])
def test_capsule_claim_changes_refuse(local_snapshot, change):
    c = deepcopy(local_snapshot.capsule)
    change(c)
    with pytest.raises(M.ModelReferenceError):
        M.validate_capsule(c)


@pytest.mark.parametrize("stamp", ["-0", "+1", "00", str(2**63), str(-(2**63) - 1), "1" * 1000])
def test_stat_time_aliases_and_overflow_are_not_accepted(local_snapshot, stamp):
    c = deepcopy(local_snapshot.capsule)
    c["files"]["README.md"]["target"]["mtime_ns"] = stamp
    with pytest.raises(M.ModelReferenceError):
        M.validate_capsule(c)


def test_changed_files_and_symlink_targets_refuse_setup(local_snapshot):
    s = local_snapshot
    path = s.root / "README.md"
    target = path.resolve()
    target.write_bytes(b"changed")
    with pytest.raises(M.ModelReferenceError):
        M.ModelBinding(s.config)
    with pytest.raises(M.ModelReferenceError):
        M.capture(s.root, run_id=RUN, packet_sha256=PACKET)


def test_capsule_short_reads_are_reassembled(local_snapshot, monkeypatch):
    original = M.os.read
    monkeypatch.setattr(M.os, "read", lambda fd, count: original(fd, min(count, 7)))
    assert M.ModelBinding(local_snapshot.config).observe(local_snapshot.config)["schema"] == M.OBSERVED_SCHEMA


def test_capsule_symlink_and_duplicate_json_refuse(local_snapshot, monkeypatch):
    s = local_snapshot
    link = s.path.with_suffix(".link")
    link.symlink_to(s.path)
    monkeypatch.setenv(M.ENV_FILE, str(link))
    with pytest.raises(M.ModelReferenceError):
        M.ModelBinding(s.config)
    monkeypatch.setenv(M.ENV_FILE, str(s.path))
    s.path.write_text('{"schema":"sdcguard.model-reference.v1","schema":"duplicate"}')
    with pytest.raises(M.ModelReferenceError):
        M.ModelBinding(s.config)


def local_profile(profile, s):
    profile.update(model=s.config.model, model_revision=M.REVISION,
                   model_reference=M.ModelBinding(s.config).observe(s.config))


def test_both_offline_profiles_preserve_actual_path(local_snapshot):
    f, a = forward_records(), assignment_records()
    for rows in (f, a):
        local_profile(rows[0]["profile"], local_snapshot)
    forward = F.audit_records(f)
    mapping = A.evaluate(a)
    assert forward["observation_status"] == "observed"
    assert forward["recorded_profile"]["model"] == local_snapshot.config.model
    assert mapping["verdict"] == "satisfied"
    assert mapping["ownership_verdict"] == mapping["physical_read_verdict"] == "unknown"
    f[0]["profile"]["model_reference"]["reference"]["files"]["README.md"]["sha256"] = "0" * 64
    a[0]["profile"]["model_reference"]["tokenizer"] = M.MODEL
    assert F.audit_records(f)["observation_status"] == "unknown"
    assert A.evaluate(a)["verdict"] == "unknown"


def test_high_bit_inode_and_signed_time_survive_record_bounds(local_snapshot):
    rows = forward_records()
    local_profile(rows[0]["profile"], local_snapshot)
    receipt = rows[0]["profile"]["model_reference"]["reference"]["files"]["README.md"]
    for index, key in enumerate(("entry", "target")):
        original = receipt[key]
        receipt[key] = M._signature(NS(st_dev=2**64 - 1, st_ino=2**63 + 1 + index,
            st_mode=original["mode"], st_size=original["size"],
            st_mtime_ns=-(2**63), st_ctime_ns=2**63 - 1))
    assert F.audit_records(rows)["observation_status"] == "observed"
    receipt["target"]["inode"] = "10000000000000000"
    assert F.audit_records(rows)["observation_status"] == "unknown"


@pytest.mark.parametrize("contradiction", ["regular_stat", "regular_path", "target_path", "target_inode", "symlink_inode"])
def test_intrinsically_conflicting_file_receipts_make_profiles_unknown(local_snapshot, contradiction):
    rows = forward_records()
    local_profile(rows[0]["profile"], local_snapshot)
    reference = rows[0]["profile"]["model_reference"]["reference"]
    first, second = reference["files"]["README.md"], reference["files"]["LICENSE"]
    if contradiction.startswith("regular"):
        first["entry"] = deepcopy(first["target"])
        if contradiction == "regular_stat":
            first["resolved_path"] = reference["resolved_snapshot"] + "/README.md"
            first["entry"]["inode"] = "0000000000000001" if first["target"]["inode"] != "0000000000000001" else "0000000000000002"
    elif contradiction == "target_path":
        second["resolved_path"] = first["resolved_path"]
    elif contradiction == "target_inode":
        second["target"].update(device=first["target"]["device"], inode=first["target"]["inode"])
    else:
        first["entry"].update(device=first["target"]["device"], inode=first["target"]["inode"])
    assert F.audit_records(rows)["observation_status"] == "unknown"
    mapping = assignment_records()
    mapping[0]["profile"].update(model=rows[0]["profile"]["model"], model_revision=M.REVISION,
                                  model_reference=deepcopy(rows[0]["profile"]["model_reference"]))
    assert A.evaluate(mapping)["verdict"] == "unknown"


def test_forward_state_uses_scoped_binding_without_engine_mutation(local_snapshot, tmp_path):
    env = environment(tmp_path / "forward")
    env.installation._end_pool(env.runner)
    local_config(env.runner, local_snapshot.config)
    env.installation._start_pool(env.runner, reset=True)
    state = env.installation.states[env.runner]
    assert state.profile["model"] == local_snapshot.config.model
    assert state.profile["model_reference"]["reference"]["model"] == M.MODEL
    assert state.model_binding is env.installation.model_binding_for(env.runner)
    env.installation.detach()


def test_mapping_header_uses_explicit_binding_and_actual_fields(local_snapshot, tmp_path):
    s = header_environment(tmp_path / "mapping")
    local_config(s.runner, local_snapshot.config)
    binding = M.ModelBinding(s.runner.model_config)
    profile, _ = R.observed_header(s.registration, s.bindings, s.participant_refs, model_binding=binding)
    assert profile["model"] == s.runner.model_config.model
    assert profile["model_reference"]["reference"]["model"] == M.MODEL
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(s.registration, s.bindings, s.participant_refs)


def test_original_canonical_profile_shape_remains_unchanged(tmp_path):
    env = environment(tmp_path)
    env.installation._start_pool(env.runner, reset=True)
    assert "model_reference" not in env.installation.states[env.runner].profile
    env.installation.detach()
    assert F.audit_records(forward_records())["observation_status"] == "observed"
    assert A.evaluate(assignment_records())["verdict"] == "satisfied"


def test_transitive_evaluator_identity_includes_binding_and_pin():
    for hashes in (forward_report.evaluator_hashes(), assignment_report.evaluator_hashes(),
                   assignment_report.evaluator_hashes(native=True)):
        assert "model_reference.py" in hashes
        assert hashes[M.PIN_RESOURCE] == M.PIN_SHA256


def test_native_header_refuses_replaced_binding_class_method(tmp_path, monkeypatch):
    from sdcguard.lifetime.forward_bridge import Registration, _freeze
    s = header_environment(tmp_path)
    owner = V.Installation(type(s.runner), tmp_path, s.env.engine, s.types, lambda: s.env.context)
    registration = Registration(owner.runner_class, _freeze(owner.engine), weakref.ref(owner))
    R.observed_header(registration, s.bindings, s.participant_refs)
    monkeypatch.setattr(V.Installation, "model_binding_for", lambda self, runner: None)
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(registration, s.bindings, s.participant_refs)


def cold_recipient(primary, folder, snapshot, monkeypatch, *, alter=None):
    """Set the actual host config before any native initialization wrapper runs."""
    import test_allocation_recipient_observer as recipient_fixture
    Path(folder).mkdir()
    original_environment = recipient_fixture.observer_environment
    def environment(source, directory, **kwargs):
        original_builder = kwargs["installation_builder"]
        def build(value, *args):
            assert value.runner.model_config is value.config.model_config
            vars(value.config.model_config).update(vars(snapshot.config))
            if alter is not None:
                original_alter = value.runner.alter
                def changed(runner):
                    original_alter(runner)
                    alter(runner)
                value.runner.alter = changed
            assert value.observer._facts == 0 and not value.observer._inits
            value.init_diagnostics = []
            original_error = value.observer._error
            def error(reason, item=None):
                active = sys.exc_info()[1]
                if active is not None:
                    value.init_diagnostics.append((reason, type(active).__name__, str(active)))
                return original_error(reason, item)
            value.observer._error = error
            installation = original_builder(value, *args)
            mapping_error = value.mapping._error
            def error_mapping(reason):
                active = sys.exc_info()[1]
                if active is not None:
                    value.init_diagnostics.append((reason, type(active).__name__, str(active)))
                return mapping_error(reason)
            value.mapping._error = error_mapping
            return installation
        kwargs["installation_builder"] = build
        return original_environment(source, directory, **kwargs)
    monkeypatch.setattr(recipient_fixture, "observer_environment", environment)
    return recipient_fixture.recipient_environment(primary, folder)


def test_local_reference_before_initialization_reaches_closed_capacity_export(primary, tmp_path,
                                                                            local_snapshot, monkeypatch):
    from sdcguard.lifetime import mapping_closure as C, allocation_recipient_recording as P
    from test_allocation_recipient_observer import allocate
    value = cold_recipient(primary, tmp_path / "cold", local_snapshot, monkeypatch)
    try:
        assert value.observer.status()["ready_associations"] == 1, json.dumps(value.init_diagnostics, indent=2)
        assert value.observer.status()["errors"] == 0
        allocate(value)
        assert value.mapping._writer is not None, json.dumps(value.init_diagnostics, indent=2)
        value.runner.shutdown()
        paired = C.capture_closed(value.journal_directory)
        raw = next(iter(paired["payload"].values()))
        records = [json.loads(line) for line in raw.splitlines()]
        header = records[0]
        assert header["profile"]["model"] == local_snapshot.config.model
        assert header["profile"]["model_reference"]["reference"]["model"] == M.MODEL
        assert {"init_begin", "runner_ready", "scheduler_ready", "acquire", "footer"} <= {r["kind"] for r in records}
        assert not any(r["kind"] == "model_entry" for r in records)
        mapping = A.evaluate(records)
        assert mapping["verdict"] == "unknown" and mapping["complete_frames"] == 0
        output = tmp_path / "recipient-packet"
        exported = P.write_capture(value.mapping.recipient, output, paired["payload"], paired["receipts"])
        report = exported["report"]
        assert report["status"] == "observed" and len(report["receipts"]) == 1
        assert report["read_rights_verdict"] == report["lease_join_verdict"] == report["physical_read_verdict"] == "unknown"
        assert P.verify(output) == {key: item for key, item in exported.items() if key != "path"}
    finally:
        value.installation.detach()


@pytest.mark.parametrize("capsule_state", ["missing", "wrong_run"])
def test_missing_capsule_before_initialization_does_not_qualify_capacity(primary, tmp_path,
                                                                       local_snapshot, monkeypatch,
                                                                       capsule_state):
    from sdcguard.lifetime import allocation_recipient_recording as P
    if capsule_state == "missing":
        monkeypatch.delenv(M.ENV_FILE)
    else:
        monkeypatch.setenv(M.ENV_RUN, "3" * 32)
    value = cold_recipient(primary, tmp_path / "missing", local_snapshot, monkeypatch)
    try:
        assert value.observer.status()["ready_associations"] == 0
        assert value.observer.status()["errors"] > 0
        assert value.mapping._writer is None
        value.runner.shutdown()
        prepared = P.prepare(value.mapping.recipient, {}, {})
        assert prepared["report"]["status"] == "unknown"
        assert prepared["report"]["scope_verdict"] == "inconclusive"
    finally:
        value.installation.detach()


def test_local_to_canonical_change_during_initialization_is_not_rebound(primary, tmp_path,
                                                                      local_snapshot, monkeypatch):
    from sdcguard.lifetime import allocation_recipient_recording as P
    def change(runner):
        runner.model_config.model = M.MODEL
    value = cold_recipient(primary, tmp_path / "changed", local_snapshot, monkeypatch, alter=change)
    try:
        assert value.observer.status()["ready_associations"] == 0
        assert value.observer.status()["errors"] > 0
        assert value.mapping._writer is None
        assert not any(kind == "scheduler_ready" for kind, fields in value.mapping._buffer)
        value.runner.shutdown()
        prepared = P.prepare(value.mapping.recipient, {}, {})
        assert prepared["report"]["status"] == "unknown"
    finally:
        value.installation.detach()


def test_initialization_registration_does_not_retain_installation(tmp_path):
    from sdcguard.lifetime.assignment_init import InitializationObserver
    from sdcguard.lifetime.forward_bridge import Registration, _freeze
    observer = InitializationObserver(lambda fact: True, NS(error=lambda reason: None))
    installation = V.Installation(type("HostRunner", (), {}), tmp_path, {}, {}, lambda: None)
    registration = Registration(installation.runner_class, _freeze(installation.engine), weakref.ref(installation))
    observer.bind_model_references(registration)
    reference = weakref.ref(installation)
    del installation
    gc.collect()
    assert reference() is None and observer._model_binding_getter() is None
    observer.detach()
