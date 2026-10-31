"""Recorder integrity, independent of the lifetime verdict implementation."""
import json
import threading
import time

from sdcguard.lifetime.adapter import JsonlSink, Probe, read_events, merge_events


def test_recording_failure_does_not_interrupt_server_and_taints_later_events():
    class Flaky:
        errors = 0
        events = []
        def emit(self, ev):
            if ev['seq'] == 1:
                raise OSError('disk full')
            self.events.append(ev)
    sink = Flaky()
    probe = Probe(sink)
    probe.alloc(('P', 0), 'x')
    probe.evidence('recording_health', healthy=False)
    assert sink.events[0]['seq'] == 2
    assert sink.events[0]['dropped'] == 1


def test_concurrent_recording_keeps_sequence_in_sink_order(tmp_path):
    sink = JsonlSink(str(tmp_path/'events.jsonl'), asynchronous=True)
    probe = Probe(sink)
    threads = [threading.Thread(target=lambda: [probe.evidence('sample') for _ in range(100)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    probe.close()
    rows = [json.loads(x) for x in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [e['seq'] for e in rows] == list(range(1, 801))
    assert all(e['dropped'] == 0 for e in rows)


def test_async_writer_failure_is_visible_in_health(tmp_path, monkeypatch):
    sink = JsonlSink(str(tmp_path/'events.jsonl'), asynchronous=True)
    original = sink._write
    def fail_once(ev):
        if ev['seq'] == 1:
            raise OSError('write failed')
        original(ev)
    monkeypatch.setattr(sink, '_write', fail_once)
    probe = Probe(sink)
    probe.evidence('sample')
    assert sink.flush()
    probe.evidence('recording_health', healthy=sink.errors == 0)
    probe.close()
    rows = read_events(str(tmp_path))
    assert rows[0]['args']['healthy'] is False
    assert rows[0]['dropped'] == 1


def test_incomplete_line_is_not_read_but_bad_complete_line_is_evidence_loss(tmp_path):
    p = tmp_path/'events.jsonl'
    p.write_bytes(b'{"kind":"evidence","args":{"what":"ok"},"t":1}\nBAD\n{"kind":')
    rows = read_events(str(tmp_path))
    assert len(rows) == 2
    assert sum(e['args']['what'] == 'recording_error' for e in rows) == 1
    assert rows[0]['args']['line'] == 2
    assert len(merge_events(rows).events) == 2


def test_missing_recording_file_is_not_clean(tmp_path):
    rows = read_events(str(tmp_path/'missing.jsonl'))
    assert rows[0]['args']['what'] == 'recording_error'


def test_same_boot_monotonic_order_ignores_wall_clock_step(tmp_path):
    rows = [dict(kind='evidence', args={}, clock_id='host:boot', mono=2., t=1., pid=1, seq=2),
            dict(kind='evidence', args={}, clock_id='host:boot', mono=1., t=5., pid=1, seq=1)]
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
    assert [e['seq'] for e in read_events(str(tmp_path))] == [1, 2]


def test_periodic_health_records_after_operation(tmp_path):
    sink = JsonlSink(str(tmp_path/'events.jsonl'))
    probe = Probe(sink, health_interval=0.01)
    probe.evidence('sample')
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        rows = read_events(str(tmp_path))
        if any(e['args'].get('what') == 'recording_health' for e in rows):
            break
        time.sleep(0.01)
    probe.close()
    sample = next(e for e in rows if e['args']['what'] == 'sample')
    health = next(e for e in rows if e['args']['what'] == 'recording_health')
    assert health['seq'] > sample['seq'] and health['mono'] >= sample['mono']
    assert health['args']['healthy'] is True and health['dropped'] == 0


def test_invalid_sort_keys_become_recording_errors(tmp_path):
    rows = [dict(kind='evidence', args={}, t=float('nan')),
            dict(kind='evidence', args={}, t=1, clock_id=[]),
            dict(kind='evidence', args={}, t=1, seq={}),
            dict(kind='evidence', args={}, t=1, mono=float('inf'))]
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
    assert all(e['args']['what'] == 'recording_error' for e in read_events(tmp_path))
