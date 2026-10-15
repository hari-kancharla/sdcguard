from sdcguard import rollback_checkpoint
from sdcguard.demo import main, run


def test_demo_catches_the_bad_device_without_false_alarms():
    res = run(out=lambda *a: None)
    assert res["ok"], res
    assert res["alert"]["step"] >= 45                       # never before the device went bad
    assert res["alert"]["rollback_to"] <= 45 - 1            # rolled back to before the first fault
    assert res["false_alarms_before"] == 0 and res["alerts_after"] == 0


def test_demo_cli_exit_code():
    assert main(["--steps", "80"]) == 0


def test_rollback_checkpoint():
    ck = [50, 100, 150, 200]
    assert rollback_checkpoint(210, ck, check_every=10) == 200
    assert rollback_checkpoint(210, ck, check_every=10, windows=2) == 150
    assert rollback_checkpoint(200, ck, check_every=50) == 150
    assert rollback_checkpoint(40, ck, check_every=10) is None
