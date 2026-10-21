"""deploy/slurm/healthcheck.sh: a failed screen drains the node, a newer passing screen returns it to service
(deploy/README.md: "A failure keeps the node drained until a new screen passes"), and a node drained for another
reason stays drained. sdcguard, scontrol and logger are stand-ins; scontrol keeps the node's state the way Slurm
shows it (`scontrol -o show node` prints Reason=<the drain reason> [user@time])."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SCONTROL = r"""#!/bin/bash
echo "$@" >> "$FAKE_DIR/scontrol.log"
if [ "$1" = "-o" ] && [ "$2" = "show" ]; then
    when="[root@2026-10-21T05:00:00]"
    echo "NodeName=$4 Arch=x86_64 State=$(cat "$FAKE_DIR/state") Reason=$(cat "$FAKE_DIR/reason") $when"
elif [ "$1" = "update" ]; then
    for a in "$@"; do
        case "$a" in
            State=DRAIN) echo "IDLE+DRAIN" > "$FAKE_DIR/state" ;;
            State=RESUME) echo "IDLE" > "$FAKE_DIR/state"; echo "(null)" > "$FAKE_DIR/reason" ;;
            Reason=*) echo "${a#Reason=}" > "$FAKE_DIR/reason" ;;
        esac
    done
fi
"""


def _node(tmp_path, state="IDLE", reason="(null)"):
    b = tmp_path / "bin"
    b.mkdir()
    (b / "sdcguard").write_text("#!/bin/sh\ncat \"$FAKE_DIR/line\"\nexit $(cat \"$FAKE_DIR/code\")\n")
    (b / "scontrol").write_text(_SCONTROL)
    (b / "logger").write_text("#!/bin/sh\nexit 0\n")
    for f in b.iterdir():
        f.chmod(0o755)
    (tmp_path / "state").write_text(state + "\n")
    (tmp_path / "reason").write_text(reason + "\n")
    return {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "FAKE_DIR": str(tmp_path),
            "SLURMD_NODENAME": "gpu-node-17", "SDCGUARD_OUT": str(tmp_path / "out")}


def _check(tmp_path, env, line, code):
    (tmp_path / "line").write_text(line)
    (tmp_path / "code").write_text(str(code))
    subprocess.run(["bash", str(ROOT / "deploy/slurm/healthcheck.sh")], env=env, check=True, timeout=60)
    return (tmp_path / "state").read_text().strip()


def test_slurm_healthcheck_returns_a_node_to_service_after_a_passing_screen(tmp_path):
    """healthcheck.sh drained on exit 1 and never resumed: after the next screen passed (exit 0) the node stayed
    drained (its reason "sdcguard: ..." is its own, which Slurm doesn't clear)."""
    env = _node(tmp_path)
    assert _check(tmp_path, env, "FAIL: 1 of 8 GPUs computed wrong results", 1) == "IDLE+DRAIN"
    assert _check(tmp_path, env, "UNKNOWN: the last screen is 40 hours old", 2) == "IDLE+DRAIN"   # not a pass
    assert _check(tmp_path, env, "OK: 8 GPUs computed correctly", 0) == "IDLE"
    calls = [c for c in (tmp_path / "scontrol.log").read_text().splitlines() if c.startswith("update")]
    assert len(calls) == 2, calls
    assert "State=DRAIN" in calls[0] and calls[-1] == "update NodeName=gpu-node-17 State=RESUME", calls


def test_slurm_healthcheck_leaves_a_node_drained_for_another_reason_alone(tmp_path):
    """A passing screen must not undo an administrator's drain (or another health check's)."""
    env = _node(tmp_path, state="IDLE+DRAIN", reason="replacing a fan")
    assert _check(tmp_path, env, "OK: 8 GPUs computed correctly", 0) == "IDLE+DRAIN"
    log = tmp_path / "scontrol.log"
    assert not any("RESUME" in c for c in (log.read_text().splitlines() if log.exists() else []))


def test_a_failed_screen_doesnt_take_over_another_drain(tmp_path):
    """Draining over an administrator's drain replaced their reason with "sdcguard: ...", so the next passing screen
    resumed a node someone else had taken out. The other reason stays (logged); once they resume the node, the
    next run drains it as its own."""
    env = _node(tmp_path, state="IDLE+DRAIN", reason="replacing a fan")
    assert _check(tmp_path, env, "FAIL: 1 of 8 GPUs computed wrong results", 1) == "IDLE+DRAIN"
    assert (tmp_path / "reason").read_text().strip() == "replacing a fan"
    assert _check(tmp_path, env, "OK: 8 GPUs computed correctly", 0) == "IDLE+DRAIN"     # never resumed
    (tmp_path / "state").write_text("IDLE\n")                                            # the admin resumes it
    (tmp_path / "reason").write_text("(null)\n")
    assert _check(tmp_path, env, "FAIL: 1 of 8 GPUs computed wrong results", 1) == "IDLE+DRAIN"
    assert (tmp_path / "reason").read_text().startswith("sdcguard: FAIL")
    assert _check(tmp_path, env, "OK: 8 GPUs computed correctly", 0) == "IDLE"
