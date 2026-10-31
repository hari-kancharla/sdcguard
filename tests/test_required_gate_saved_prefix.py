"""Replay retained ordinary traffic only. Owner: Hari Krishna Kancharla."""
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from sdcguard import cli, evidence_io as io, evidence_json_budget as budget, required_gate as gate


SAVED = Path(__file__).parent / 'data' / 'checksum_owner_3a2ad847'


@pytest.fixture(scope='module')
def saved_run():
    receipt = json.loads((SAVED / 'SOURCE.json').read_bytes())
    for name, expected in receipt['files'].items():
        raw = (SAVED / name).read_bytes()
        assert {'bytes': len(raw), 'sha256': io.digest(raw)} == expected
    result = tuple((SAVED / name).read_bytes() for name in
                   ('RESULT.json', 'ordinary-evidence.zip', 'CPU_PREFLIGHT.json'))
    assert io.digest(result[1]) == 'f0b2387e9dcdfaedec5987e433676229ec2ab7083d49e890a9cc5d1dbd99127d'
    return result


@pytest.fixture(scope='module')
def replay(saved_run):
    return gate.checksum_gate(*saved_run, ['checksum'])


def test_retained_prefix_run_replays_without_dropping_any_request(replay):
    assert replay['decision'] == 'HOLD'
    assert replay['selected_requests'] == replay['total_planned_requests'] == 153
    assert replay['excluded_requests'] == 0
    assert replay['passed_requests'] == 9
    assert replay['hold_reasons'] == ['required_checksum_coverage_incomplete']
    assert Counter(row['decision'] for row in replay['requests']) == {'PASS': 9, 'HOLD': 144}
    assert len({row['client_request_id'] for row in replay['requests']}) == 153
    assert {row['slot'] for row in replay['requests'] if row['decision'] == 'PASS'} == set(range(0, 153, 17))
    assert replay['runtime_qualified'] is False
    assert replay['combined_checks_supported'] is False
    for row in replay['requests']:
        check = row['checks']['checksum']
        assert check['native_result'] == 'passed'
        if row['decision'] == 'PASS':
            assert row['reasons'] == []
            assert check['completed'] is True and check['evidence_complete'] is True
            assert check['expected_units'] == check['compared_units']
        else:
            assert row['reasons'] == ['cached prefix not covered by checksum',
                                      'checksum:cached prefix not covered by checksum']
            assert check['evidence_complete'] is False
            assert set(check['compared_units']) < set(check['expected_units'])


def test_retained_prefix_token_counts_match_every_saved_join_row(saved_run, replay):
    files = gate.read_archive(saved_run[1])
    recorded = json.loads(files['CHECKSUM_JOIN.json'])
    original = {row['client_request_id']: row['token_coverage'] for row in recorded['requests']}
    assert recorded['summary']['complete_client_requests'] == 153
    assert recorded['summary']['native_pass_requests'] == 153
    totals = Counter()
    for row in replay['requests']:
        tokens = row['token_coverage']
        assert tokens == original[row['client_request_id']]
        assert tokens['prompt_tokens'] == tokens['transferred_tokens_checked'] + tokens['decode_cached_tokens_unchecked']
        assert tokens['decode_locally_computed_tokens'] == 0
        if row['decision'] == 'PASS':
            assert tokens['decode_cached_tokens_unchecked'] == 0
            assert tokens['prefill_reused_cached_blocks'] is False
        else:
            assert tokens['decode_cached_tokens_unchecked'] > 0
            assert tokens['transferred_tokens_checked'] > 0
            assert tokens['prefill_reused_cached_blocks'] is True
        totals.update({key: value for key, value in tokens.items() if type(value) is int})
    assert dict(totals) == {'prompt_tokens': 109790, 'transferred_tokens_checked': 9822,
                           'decode_cached_tokens_unchecked': 99968, 'decode_locally_computed_tokens': 0,
                           'prefill_cached_tokens': 99968}


def test_saved_recording_reproduces_old_aggregate_budget_failure(saved_run, monkeypatch):
    monkeypatch.setattr(budget, 'MAX_TOTAL_JSON_NODES', io.MAX_JSON_NODES)
    with pytest.raises(io.EvidenceError, match='raw structured-input node allowance exceeded'):
        gate.checksum_gate(*saved_run, ['checksum'])


def test_saved_recording_still_refuses_over_budget_structured_members(saved_run):
    result_raw, archive_raw, preflight_raw = saved_run
    files = gate.read_archive(archive_raw)
    # Each added document is below the per-file cap. Together they exceed the
    # aggregate cap, and must be refused before semantic reconstruction.
    for name in ('cache-budget-a.json', 'cache-budget-b.json'):
        files['observer-on/decode/' + name] = b'[' + b'0,' * 499999 + b'0]'
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    edited = stream.getvalue()
    result = json.loads(result_raw)
    result['archive_sha256'] = io.digest(edited)
    with pytest.raises(io.EvidenceError, match='raw structured-input node allowance exceeded'):
        gate.checksum_gate(io.json_bytes(result), edited, preflight_raw, ['checksum'])


def test_global_build_error_still_holds_even_the_nine_complete_requests(saved_run):
    result, archive, preflight = saved_run
    value = json.loads(preflight)
    value['installed_engine']['versions']['vllm'] = 'unknown'
    answer = gate.checksum_gate(result, archive, io.json_bytes(value), ['checksum'])
    assert answer['decision'] == 'HOLD' and answer['passed_requests'] == 0
    assert 'unsupported_recorded_engine_version' in answer['hold_reasons']
    assert all(row['decision'] == 'HOLD' and 'unsupported_recorded_engine_version' in row['reasons']
               for row in answer['requests'])


def test_cli_saved_prefix_hold_is_a_policy_result_not_a_parser_error(saved_run, capsys):
    assert cli.main(['gate', str(SAVED), '--require', 'checksum', '--json']) == 2
    answer = json.loads(capsys.readouterr().out)
    assert answer['schema'] == 'sdcguard.required-check-gate.v1'
    assert answer['passed_requests'] == 9 and answer['selected_requests'] == 153
    assert answer['hold_reasons'] == ['required_checksum_coverage_incomplete']
