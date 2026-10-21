"""Every sdcguard command exits 1 only for wrong results: bad input, or a part that can't run, is 2 (couldn't
tell), also for the subcommands with parsers of their own (check, board, demo, selftest, screen, casefile, blas,
issues), which cli.main used to dispatch outside the handler that turns an error into exit 2."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))


@pytest.mark.parametrize("argv", [["blas", "--shape", "4096x4096"], ["screen", "--device", "bogus"],
                                  ["board", "--issues", "someone/somewhere"]])
def test_bad_input_exits_2_not_1(argv, tmp_path):
    """`sdcguard blas --shape 4096x4096` (ValueError from parse_shape), `sdcguard screen --device bogus` (torch.device
    raised before screen.main's try) and `sdcguard board --issues` without the gh CLI died with a traceback and exit
    1, which blas and screen document as wrong results."""
    cmd = [sys.executable, "-m", "sdcguard", *argv] + (["--out", str(tmp_path / "out")] if argv[0] != "blas" else [])
    env = dict(os.environ, PYTHONPATH=SRC)
    if argv[0] == "board":
        env["PATH"] = str(tmp_path)                  # no gh here
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300, cwd=str(tmp_path))
    assert p.returncode == 2, (p.returncode, p.stdout[-300:], p.stderr[-300:])
    assert "Traceback" not in p.stderr, p.stderr[-300:]
