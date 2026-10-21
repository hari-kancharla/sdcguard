"""The board bot's comments (board.py): the reason a posted card can't go on the board is posted as a comment
from the repository's account, so it never repeats a name from the card as it was written (no mentions, links or
long text; _shown() is there for that)."""
import os

from sdcguard import board

HERE = os.path.dirname(os.path.abspath(__file__))
EVIL = "@everyone [fix](https://evil.example/x)"


def _card(**changes):
    import json
    with open(os.path.join(HERE, "fixtures", "board_card.json")) as f:
        r = json.load(f)
    r.update(changes)
    r.pop("digest", None)
    r["digest"] = board.digest(r)
    return r


def _safe(why):
    return why and "@everyone" not in why and "](" not in why and "evil.example" not in why


def test_an_unknown_field_name_is_never_repeated_verbatim():
    r = _card()
    r[EVIL] = 1
    r.pop("digest")
    r["digest"] = board.digest(r)
    why = board.validate(r)
    assert _safe(why) and why.startswith("malformed card (fields the check doesn't write"), why


def test_a_math_mode_name_is_never_repeated_verbatim():
    g = dict(_card()["gpu"])
    g.pop("compute_cap")
    r = _card(gpu=g, wrong={**_card()["wrong"], "math_modes": 0},
              math_modes={"ran": ["bf16"], "matmuls": 9, "wrong": 0, "not_checked": {EVIL: "x"}, "errors": {},
                          "typical_error_caught": {}})
    why = board.validate(r)
    assert _safe(why) and "math mode didn't run" in why, why
