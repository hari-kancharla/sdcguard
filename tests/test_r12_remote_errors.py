"""Worker exceptions must not look like an unfinished result poll."""
import pytest

from _r12_runner_helpers import load_runner


@pytest.mark.parametrize("error", [TimeoutError("worker stage timed out"), RuntimeError("worker failed")])
def test_worker_failure_is_returned_as_a_terminal_error_record(monkeypatch, error):
    # These decorators are local stubs; no SDK client or remote function is invoked.
    m = load_runner("modal_check")

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(m, "_tests", fail)
    r = m.tests()
    assert r["run"] != 0
    assert type(error).__name__ in r["text"]
    assert str(error) in r["text"]
