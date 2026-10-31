"""Portable checksum client evidence. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from sdcguard import cli, checksum_client as client, checksum_client_report as report
from sdcguard import evidence_io as io
from test_checksum_client import case_bytes, client_case


def inputs(tmp_path, values=None, case='complete'):
    directory = tmp_path / 'raw'
    directory.mkdir()
    raws = case_bytes(case) if values is None else tuple(io.json_bytes(v) for v in values)
    paths = []
    for name, raw in zip(report.INPUTS, raws):
        path = directory / name
        path.write_bytes(raw)
        paths.append(path)
    return directory, paths, raws


@pytest.mark.parametrize('case,reason', [('complete', None), ('partial-span', 'partial_checksum_span'),
                                         ('empty-replicas', 'empty_replica_inputs')])
def test_installed_command_shape_retains_and_recomputes_each_client(case, reason, tmp_path, capsys):
    directory, paths, raws = inputs(tmp_path, case=case)
    packet = tmp_path / 'packet'
    arguments = ['--plan', str(paths[0]), '--inventory', str(paths[1]), '--captures', str(paths[2])]
    assert cli.main(['checksum-client', 'inspect', *arguments, '--json']) == 2
    observed = json.loads(capsys.readouterr().out)
    assert observed == client.evaluate(*raws)
    assert observed['release_decision'] == 'HOLD' and observed['runtime_qualified'] is False
    assert observed['summary']['declared_requests'] == 2
    assert observed['summary']['on_requests'] == observed['summary']['off_requests'] == 1
    assert observed['client_inventory']['status'] == 'complete'
    assert cli.main(['checksum-client', 'capture', *arguments, '--out', str(packet), '--json']) == 2
    captured = json.loads(capsys.readouterr().out)
    assert captured == observed
    for name, raw in zip(report.INPUTS, raws):
        assert (packet / 'inputs' / name).read_bytes() == raw
    shutil.rmtree(directory)
    assert cli.main(['checksum-client', 'verify', str(packet), '--json']) == 2
    assert json.loads(capsys.readouterr().out) == observed
    assert cli.main(['checksum-client', 'verify', str(packet)]) == 2
    text = capsys.readouterr().out
    assert report.TITLE in text and 'Release: HOLD' in text and 'observer_disabled' in text
    assert (packet / 'report.txt').read_text() == text
    assert json.loads((packet / 'report.json').read_bytes()) == observed
    assert report.TITLE in (packet / 'report.html').read_text()
    if reason:
        assert 'Engine pass, not covered: ' + reason in text
        assert observed['summary']['verified_on_requests'] == 0
    else:
        assert observed['summary']['verified_on_requests'] == 1


def test_missing_client_receipt_stays_in_portable_report_denominator(tmp_path):
    values = client_case()
    values[1]['requests'] = values[1]['requests'][:1]
    _, paths, _ = inputs(tmp_path, values)
    packet = tmp_path / 'packet'
    observed = report.capture(*paths, packet)
    assert observed['summary']['declared_requests'] == 2
    assert observed['client_inventory']['status'] == 'unknown'
    assert observed['summary']['verified_on_requests'] == 0
    assert report.verify(packet) == observed


@pytest.mark.parametrize('kind', ['derived_report', 'raw_inventory', 'source_profile'])
def test_rehashed_packet_edits_do_not_replace_recomputation(tmp_path, kind):
    _, paths, _ = inputs(tmp_path)
    packet = tmp_path / 'packet'
    report.capture(*paths, packet)
    relative = {'derived_report': 'report.json', 'raw_inventory': 'inputs/CLIENT_INVENTORY.json',
                'source_profile': 'source-profile.json'}[kind]
    path = packet / relative
    value = json.loads(path.read_bytes())
    if kind == 'derived_report':
        value['release_decision'] = 'PASS'
        value['summary']['verified_on_requests'] = 2
    elif kind == 'raw_inventory':
        value['requests'] = []
    else:
        value['revision'] = '0' * 40
    path.write_bytes(io.json_bytes(value))
    manifest_path = packet / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['files'][relative] = dict(bytes=path.stat().st_size, sha256=io.digest(path.read_bytes()))
    manifest_path.write_bytes(io.json_bytes(manifest))
    with pytest.raises(io.EvidenceError):
        report.verify(packet)


def test_output_cannot_overlap_independent_raw_inputs(tmp_path):
    directory, paths, _ = inputs(tmp_path)
    with pytest.raises(io.EvidenceError):
        report.capture(*paths, directory)
    assert all(path.exists() for path in paths)


def test_cli_missing_input_returns_inconclusive_without_traceback(tmp_path, capsys):
    missing = str(tmp_path / 'missing.json')
    assert cli.main(['checksum-client', 'inspect', '--plan', missing, '--inventory', missing,
                     '--captures', missing]) == 2
    text = capsys.readouterr().out
    assert 'Client checksum evidence is inconclusive:' in text
    assert 'Traceback' not in text
