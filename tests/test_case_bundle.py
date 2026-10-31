"""A clean-installed package carries inert, pinned regression artifacts."""
from copy import deepcopy
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit

import pytest

from sdcguard.case_bundle import export_bundle, payloads, verify_bundle
from sdcguard.regression import historical_case, judge, main


def test_default_export_is_complete_without_a_checkout_and_is_movable(tmp_path):
    packet = tmp_path / 'packet'
    export_bundle(packet)
    case = historical_case()
    for label, variant in [('stock', 'unpatched'), ('fixed', 'fixed')]:
        value = json.loads((packet / (label + '_RESULT.json')).read_text())
        report = judge(value, variant)
        assert report['status'] == 'pass'
        assert report['scope'].startswith('Historical experiment criterion only')
    assert json.loads((packet / 'candidate.json').read_text()) == case['candidate']
    assert hashlib.sha256((packet / 'existing_case.py').read_bytes()).hexdigest() == case['reviewed_harness_sha256']
    assert 'Minimality is NOT established' in (packet / 'README.md').read_text()
    assert verify_bundle(packet)['status'] == 'verified'
    shutil.move(packet, tmp_path / 'moved')
    report = verify_bundle(tmp_path / 'moved')
    assert report['status'] == 'verified' and report['gpu_executed'] is False
    assert report['historical_model_revision'] == 'unresolved'
    assert 'not a signature' in report['trust']


@pytest.mark.parametrize('move', [False, True])
def test_exported_result_links_resolve_to_exact_saved_artifacts(tmp_path, move):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hrefs = []

        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                self.hrefs.extend(value for name, value in attrs if name == 'href')

    packet = tmp_path / 'packet'
    export_bundle(packet)
    if move:
        shutil.move(packet, tmp_path / 'relocated')
        packet = tmp_path / 'relocated'
    links = Links()
    links.feed((packet / 'RESULT.html').read_text())
    artifacts = historical_case()['artifacts']
    assert len(links.hrefs) == len(artifacts)
    resolved_names = set()
    for href in links.hrefs:
        url = urlsplit(href)
        assert not any((url.scheme, url.netloc, url.query, url.fragment))
        target = (packet / unquote(url.path)).resolve()
        assert target.parent == packet.resolve(), 'artifact link must stay inside the portable packet'
        assert target.is_file(), 'result page artifact link must open a real exported file'
        assert hashlib.sha256(target.read_bytes()).hexdigest() == artifacts[target.name]['sha256']
        resolved_names.add(target.name)
    assert resolved_names == set(artifacts)


@pytest.mark.parametrize('name', ['existing_case.py', 'candidate.json', 'case.json', 'stock_RESULT.json',
                                  'fixed_RESULT.json', 'README.md', 'SUPPORT.md', 'run.py'])
def test_changed_files_are_rejected_even_if_manifest_is_rehashed(tmp_path, name):
    packet = tmp_path / 'packet'
    export_bundle(packet)
    changed = (packet / name).read_bytes() + b'\nchanged\n'
    (packet / name).write_bytes(changed)
    path = packet / 'BUNDLE_MANIFEST.json'
    manifest = json.loads(path.read_text())
    manifest['file_sha256'][name] = hashlib.sha256(changed).hexdigest()
    path.write_text(json.dumps(manifest))
    assert verify_bundle(packet)['status'] == 'inconclusive'


def test_missing_extra_and_symlink_files_are_not_verified(tmp_path):
    for kind in ['missing', 'extra', 'symlink']:
        packet = tmp_path / kind
        export_bundle(packet)
        if kind == 'missing':
            (packet / 'run.py').unlink()
        elif kind == 'extra':
            (packet / 'unlisted.txt').write_text('not in packet')
        else:
            # Link to an identical owned fixture still fails: a portable packet has regular files.
            copy = tmp_path / 'owned-copy'
            copy.write_bytes((packet / 'run.py').read_bytes())
            (packet / 'run.py').unlink()
            (packet / 'run.py').symlink_to(copy)
        assert verify_bundle(packet)['status'] == 'inconclusive'


def test_export_refuses_existing_directory_without_changing_it(tmp_path):
    packet = tmp_path / 'packet'
    packet.mkdir()
    sentinel = packet / 'mine'
    sentinel.write_text('preserve')
    with pytest.raises(ValueError, match='exists'):
        export_bundle(packet)
    assert sentinel.read_text() == 'preserve' and list(packet.iterdir()) == [sentinel]


def test_cli_exports_and_verifies_without_executing_a_runner(tmp_path, monkeypatch, capsys):
    import sdcguard.regression as regression
    monkeypatch.setattr(regression.subprocess, 'call', lambda *a, **k: pytest.fail('no execution during export or verify'))
    assert main(['export', str(tmp_path / 'packet')]) == 0
    assert main(['verify', str(tmp_path / 'packet')]) == 0
    assert '"status": "verified"' in capsys.readouterr().out


def test_absent_and_malformed_manifest_are_inconclusive(tmp_path):
    assert verify_bundle(tmp_path / 'missing')['status'] == 'inconclusive'
    packet = tmp_path / 'packet'
    export_bundle(packet)
    (packet / 'BUNDLE_MANIFEST.json').write_text('[]')
    assert verify_bundle(packet)['status'] == 'inconclusive'


def test_packaged_runner_matches_reviewed_checkout_source():
    checkout = Path(__file__).resolve().parents[1]
    assert payloads()['existing_case.py'] == (checkout / 'experiments/lifetime_pd_check.py').read_bytes()


def test_fresh_regression_requires_exact_model_and_tokenizer_pin():
    from test_scoped_61229 import fresh_result
    from sdcguard.regression import digest
    for field in ('model_revision', 'tokenizer_revision'):
        result, manifest = fresh_result()
        result[field] = 'b' * 40
        manifest['result_canonical_sha256'] = digest(result)
        assert judge(result, 'fixed', manifest)['status'] == 'inconclusive'
    result, manifest = fresh_result()
    manifest['model_pin']['files']['config.json'] = 'c' * 64
    assert judge(result, 'fixed', manifest)['status'] == 'inconclusive'


def test_harness_defaults_to_pinned_model_and_tokenizer_without_starting_server(tmp_path, monkeypatch):
    from test_lifetime_pd_check import X
    observed = []
    def stop_before_server(cmd, **kwargs):
        observed.append(cmd)
        raise RuntimeError('owned test intercepts startup; no server runs')
    monkeypatch.setattr(X.subprocess, 'Popen', stop_before_server)
    assert X.main(['--out', str(tmp_path / 'run')]) == 4
    command = observed[0]
    pin = historical_case()['replay_model_revision']
    assert command[command.index('--revision') + 1] == pin
    assert command[command.index('--tokenizer-revision') + 1] == pin
    result = json.loads((tmp_path / 'run/RESULT.json').read_text())
    assert result['model_revision'] == result['tokenizer_revision'] == pin


def test_floating_model_revision_is_rejected_before_output_or_server(tmp_path):
    from test_lifetime_pd_check import X
    with pytest.raises(SystemExit) as exc:
        X.main(['--out', str(tmp_path / 'run'), '--revision', 'main'])
    assert exc.value.code == 2 and not (tmp_path / 'run').exists()


def test_mutation_of_an_earlier_file_during_final_read_never_verifies(tmp_path, monkeypatch):
    import os
    import sdcguard.case_bundle as bundle
    packet = tmp_path / 'raced'
    export_bundle(packet)
    final_inode = (packet / 'stock_STATUS.json').stat().st_ino
    read = os.read
    changed = False
    def racing_read(fd, n):
        nonlocal changed
        if not changed and os.fstat(fd).st_ino == final_inode:
            changed = True
            (packet / 'run.py').write_text('changed during verification\n')
        return read(fd, n)
    monkeypatch.setattr(bundle.os, 'read', racing_read)
    result = verify_bundle(packet)
    assert changed and result['status'] == 'inconclusive'
    assert 'changed during verification' in result['missing_or_changed'][0]


def test_snapshot_hashing_does_not_read_whole_weight_file(tmp_path, monkeypatch):
    from sdcguard.regression import _model_files
    data = b'owned fixture' * 1000
    (tmp_path / 'weights.bin').write_bytes(data)
    def forbidden(self):
        raise AssertionError('whole-file allocation is not allowed for snapshot hashing')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    assert _model_files(tmp_path) == {'weights.bin': hashlib.sha256(data).hexdigest()}
