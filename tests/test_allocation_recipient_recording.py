"""Inert portable recipient inventory and single-export controls.

Owner: Hari Krishna Kancharla.
These controls use ordinary metadata fixtures. No engine body, inference
request, scheduling condition, server, device or transfer is executed.
"""
from copy import deepcopy
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from sdcguard.lifetime import allocation_recipient_recording as P
from sdcguard.lifetime.allocation_recipient_observer import AllocationRecipientObserver
from test_allocation_recipient_obligations import example


def observer_fixture():
    records, payload, receipts, _ = example()
    class Mapping:
        pass
    mapping = Mapping()
    mapping.registry = SimpleNamespace(new_token=lambda: records[0]['session'])
    observer = AllocationRecipientObserver(mapping)
    footer = records[-1]
    observer.records = deepcopy(records[:-1])
    observer._header = observer.records[0]
    observer._closed = True
    for field in ('outer_calls', 'phase_calls', 'pool_returns', 'block_facts', 'empty_pool_returns'):
        setattr(observer, '_' + field, footer[field])
    return observer, mapping, payload, receipts


def packet(tmp_path):
    observer, mapping, payload, receipts = observer_fixture()
    directory = tmp_path / 'packet'
    result = P.write_capture(observer, directory, payload, receipts)
    assert result['report']['status'] == 'observed'
    return directory, observer, mapping


def test_selective_inventory_never_uses_eager_listing_or_unlisted_content(tmp_path, monkeypatch):
    directory, _, retained_mapping = packet(tmp_path)
    manifest = P.io.parse_json((directory / P.MANIFEST).read_bytes())
    expected = set(manifest['files']) | {P.MANIFEST}
    original = P.config_inputs._read_regular
    reads = []
    def read(root, parts, limit):
        reads.append('/'.join(parts))
        return original(root, parts, limit)
    def eager(*args, **kwargs):
        raise AssertionError('eager external listing is outside the recipient contract')
    monkeypatch.setattr(P.config_inputs, '_read_regular', read)
    monkeypatch.setattr(P.io, 'names', eager)
    monkeypatch.setattr(P.os, 'listdir', eager)
    assert P.verify(directory)['report']['status'] == 'observed'
    assert set(reads) == expected
    assert retained_mapping is not None


def test_late_unlisted_empty_directory_is_refused_after_semantic_replay(tmp_path, monkeypatch):
    directory, _, retained_mapping = packet(tmp_path)
    original = P.contract.evaluate
    def evaluate(*args, **kwargs):
        (directory / 'unlisted-empty').mkdir()
        return original(*args, **kwargs)
    monkeypatch.setattr(P.contract, 'evaluate', evaluate)
    with pytest.raises(ValueError, match='unlisted directory'):
        P.verify(directory)
    assert (directory / 'unlisted-empty').is_dir()
    assert retained_mapping is not None


def test_directory_added_during_last_byte_recheck_is_refused(tmp_path, monkeypatch):
    directory, _, retained_mapping = packet(tmp_path)
    original = P.config_inputs._read_regular
    counts = {}
    inserted = []
    def read(root, parts, limit):
        name = '/'.join(parts)
        raw = original(root, parts, limit)
        counts[name] = counts.get(name, 0) + 1
        if name == 'recipient.jsonl' and counts[name] == 2:
            (directory / 'late-empty').mkdir()
            inserted.append(True)
        return raw
    monkeypatch.setattr(P.config_inputs, '_read_regular', read)
    with pytest.raises(ValueError, match='unlisted directory'):
        P.verify(directory)
    assert inserted == [True]
    assert retained_mapping is not None


def test_unknown_tree_is_refused_without_opening_or_collecting_its_contents(tmp_path, monkeypatch):
    directory, _, retained_mapping = packet(tmp_path)
    unrelated = directory / 'unlisted-tree'
    unrelated.mkdir()
    (unrelated / 'never-read').write_bytes(b'ordinary unrelated fixture')
    original_open, original_read = P.os.open, P.config_inputs._read_regular
    opened, read_names = [], []
    def opened_file(name, *args, **kwargs):
        opened.append(str(name))
        assert str(name) not in ('unlisted-tree', 'never-read')
        return original_open(name, *args, **kwargs)
    def read(root, parts, limit):
        read_names.append('/'.join(parts))
        return original_read(root, parts, limit)
    monkeypatch.setattr(P.os, 'open', opened_file)
    monkeypatch.setattr(P.config_inputs, '_read_regular', read)
    with pytest.raises(ValueError, match='unlisted directory'):
        P.verify(directory)
    assert read_names == [P.MANIFEST]
    assert not any(name.endswith('never-read') for name in opened)
    assert retained_mapping is not None


def test_streaming_count_refusal_does_not_materialize_external_list(tmp_path, monkeypatch):
    directory, _, retained_mapping = packet(tmp_path)
    for number in range(6):
        (directory / ('extra' + str(number))).write_bytes(b'')
    original = P.os.scandir
    counts = []
    class Stream:
        def __init__(self, *args, **kwargs):
            self.stream = original(*args, **kwargs)
            self.count = 0
        def __enter__(self):
            self.stream.__enter__()
            counts.append(self)
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def __iter__(self):
            for entry in self.stream:
                self.count += 1
                yield entry
    def eager(*args, **kwargs):
        raise AssertionError('external names must stream')
    monkeypatch.setattr(P.io, 'MAX_FILES', 4)
    monkeypatch.setattr(P.os, 'scandir', Stream)
    monkeypatch.setattr(P.os, 'listdir', eager)
    with pytest.raises(ValueError):
        P.verify(directory)
    assert counts and sum(stream.count for stream in counts) <= 6
    assert retained_mapping is not None


def test_declared_aggregate_bound_is_checked_before_listed_content_reads(tmp_path, monkeypatch):
    directory, _, retained_mapping = packet(tmp_path)
    path = directory / P.MANIFEST
    manifest = P.io.parse_json(path.read_bytes())
    for name, row in manifest['files'].items():
        if name.startswith('mapping/'):
            row['bytes'] = P.io.MAX_FILE_BYTES
    path.write_bytes(P.io.json_bytes(manifest))
    original = P.config_inputs._read_regular
    reads = []
    def read(root, parts, limit):
        reads.append('/'.join(parts))
        return original(root, parts, limit)
    monkeypatch.setattr(P.config_inputs, '_read_regular', read)
    with pytest.raises(ValueError, match='aggregate'):
        P.verify(directory)
    assert reads == [P.MANIFEST]
    assert retained_mapping is not None


def test_recipient_physical_bound_is_checked_before_content_reads(tmp_path, monkeypatch):
    directory, _, retained_mapping = packet(tmp_path)
    original = P.config_inputs._read_regular
    reads = []
    def read(root, parts, limit):
        reads.append('/'.join(parts))
        return original(root, parts, limit)
    monkeypatch.setattr(P.config_inputs, '_read_regular', read)
    monkeypatch.setattr(P.contract, 'MAX_TOTAL_BYTES', 2)
    with pytest.raises(ValueError, match='journal exceeds'):
        P.verify(directory)
    assert reads == [P.MANIFEST]
    assert retained_mapping is not None


def test_listed_reads_use_declared_length_and_remaining_aggregate_allowance(tmp_path, monkeypatch):
    directory, _, retained_mapping = packet(tmp_path)
    manifest = P.io.parse_json((directory / P.MANIFEST).read_bytes())
    original = P.config_inputs._read_regular
    reads = []
    def read(root, parts, limit):
        raw = original(root, parts, limit)
        reads.append(('/'.join(parts), limit, len(raw)))
        return raw
    monkeypatch.setattr(P.config_inputs, '_read_regular', read)
    assert P.verify(directory)['report']['status'] == 'observed'
    first_reads = [row for row in reads[1:1 + len(manifest['files'])]]
    remaining = P.contract.MAX_TOTAL_BYTES + 64 * 1024 * 1024
    for name, limit, actual in first_reads:
        assert name in manifest['files']
        assert limit == min(manifest['files'][name]['bytes'], remaining)
        remaining -= actual
    assert retained_mapping is not None


@pytest.mark.parametrize('kind', ['link', 'fifo'])
def test_nonregular_listed_member_refuses_without_opening_contents(tmp_path, monkeypatch, kind):
    directory, _, retained_mapping = packet(tmp_path)
    path = directory / 'recipient.jsonl'
    path.unlink()
    if kind == 'link':
        path.symlink_to(tmp_path / 'outside-unread')
    else:
        os.mkfifo(path)
    original = P.config_inputs._read_regular
    reads = []
    def read(root, parts, limit):
        reads.append('/'.join(parts))
        return original(root, parts, limit)
    monkeypatch.setattr(P.config_inputs, '_read_regular', read)
    with pytest.raises(ValueError, match='regular file'):
        P.verify(directory)
    assert reads == [P.MANIFEST]
    assert retained_mapping is not None


def test_one_export_is_reserved_before_an_interleaved_second_caller(tmp_path, monkeypatch):
    observer, retained_mapping, payload, receipts = observer_fixture()
    original = P.prepare
    entered, release = threading.Event(), threading.Event()
    results, failures = [], []
    def prepare(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    def first():
        try:
            results.append(P.write_capture(observer, tmp_path / 'first', payload, receipts))
        except BaseException as exc:
            failures.append(exc)
    monkeypatch.setattr(P, 'prepare', prepare)
    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert entered.wait(3)
        assert observer.status['export_attempted'] is True
        assert observer.status['export_in_progress'] is True
        assert observer.status['exported'] is False
        with pytest.raises(ValueError, match='already reserved'):
            P.write_capture(observer, tmp_path / 'second', payload, receipts)
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and not failures and len(results) == 1
    assert results[0]['report']['status'] == 'observed'
    assert observer.status['exported'] is True and observer.status['export_in_progress'] is False
    assert (tmp_path / 'first' / P.MANIFEST).is_file()
    assert not (tmp_path / 'second').exists()
    assert retained_mapping is not None


def test_nested_export_cannot_cross_the_admission_guard(tmp_path, monkeypatch):
    observer, retained_mapping, payload, receipts = observer_fixture()
    original = P.prepare
    refusals = []
    def prepare(*args, **kwargs):
        with pytest.raises(ValueError, match='already reserved'):
            P.write_capture(observer, tmp_path / 'nested', payload, receipts)
        refusals.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(P, 'prepare', prepare)
    result = P.write_capture(observer, tmp_path / 'outer', payload, receipts)
    assert result['report']['status'] == 'observed' and refusals == [True]
    assert not (tmp_path / 'nested').exists()
    assert retained_mapping is not None


def test_failed_attempt_is_consumed_without_overwriting_existing_output(tmp_path):
    observer, retained_mapping, payload, receipts = observer_fixture()
    existing = tmp_path / 'existing'
    existing.mkdir()
    (existing / 'keep').write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        P.write_capture(observer, existing, payload, receipts)
    assert observer.status['export_attempted'] is True
    assert observer.status['export_in_progress'] is False
    assert observer.status['exported'] is False
    with pytest.raises(ValueError, match='already reserved'):
        P.write_capture(observer, tmp_path / 'another', payload, receipts)
    assert (existing / 'keep').read_bytes() == b'preserve'
    assert not (tmp_path / 'another').exists()
    assert retained_mapping is not None


def test_interrupted_write_keeps_partial_files_and_consumes_attempt(tmp_path, monkeypatch):
    observer, retained_mapping, payload, receipts = observer_fixture()
    original = P.io.write_file
    writes = []
    def write(root, name, raw):
        if writes:
            raise OSError('ordinary local write interruption')
        original(root, name, raw)
        writes.append(name)
    monkeypatch.setattr(P.io, 'write_file', write)
    with pytest.raises(OSError, match='interruption'):
        P.write_capture(observer, tmp_path / 'partial', payload, receipts)
    assert writes and (tmp_path / 'partial' / writes[0]).is_file()
    assert not (tmp_path / 'partial' / P.MANIFEST).exists()
    assert observer.status['export_attempted'] is True
    assert observer.status['exported'] is False and observer.status['export_in_progress'] is False
    with pytest.raises(ValueError, match='already reserved'):
        P.write_capture(observer, tmp_path / 'another', payload, receipts)
    assert not (tmp_path / 'another').exists()
    assert retained_mapping is not None
