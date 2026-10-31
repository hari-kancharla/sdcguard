"""Admit raw structured bytes under one shared lexical node allowance.

Owner: Hari Krishna Kancharla.
This scan builds no JSON tree and supplies no semantic validity or verdict.
Strict supported JSON has one node per container, string including object key,
and primitive. Malformed or duplicate-key input is counted lexically and still
needs its existing strict verifier. Byte, file and depth bounds remain separate.
"""
from . import evidence_io as io


# Aggregate admission can include the same client receipt in both its journal
# and assembled inventory. Individual JSON trees retain io.MAX_JSON_NODES.
MAX_TOTAL_JSON_NODES = 2_000_000

_BOMS = (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")
_SPACE = (9, 10, 13, 32)
_BOUNDARY = (44, 58, 93, 125)  # comma, colon and closing containers


def _encoding_error():
    raise io.EvidenceError("unsupported structured encoding; NUL-free UTF8 without a leading BOM is required")


def _count_member(raw, jsonl, allowance):
    nodes = 0
    string = escaped = atom = False
    record_start = True
    needed, lower, upper = 0, 0x80, 0xBF

    def charge():
        nonlocal nodes
        nodes += 1
        if nodes > allowance:
            raise io.EvidenceError("raw structured-input node allowance exceeded")

    for index, byte in enumerate(raw):
        if record_start and raw.startswith(_BOMS, index):
            _encoding_error()
        if byte == 0:
            _encoding_error()
        # Incremental scalar-value UTF8 validation, including overlong, surrogate
        # and above-U+10FFFF exclusions. No decoded string is allocated.
        if needed:
            if not lower <= byte <= upper:
                _encoding_error()
            needed -= 1
            lower, upper = 0x80, 0xBF
        elif byte < 0x80:
            pass
        elif 0xC2 <= byte <= 0xDF:
            needed = 1
        elif 0xE0 <= byte <= 0xEF:
            needed = 2
            lower = 0xA0 if byte == 0xE0 else 0x80
            upper = 0x9F if byte == 0xED else 0xBF
        elif 0xF0 <= byte <= 0xF4:
            needed = 3
            lower = 0x90 if byte == 0xF0 else 0x80
            upper = 0x8F if byte == 0xF4 else 0xBF
        else:
            _encoding_error()

        if jsonl and byte in (10, 13):
            # bytes.splitlines readers parse physical CR/LF records separately.
            # A malformed quote or escape must not hide later parsed records.
            string = escaped = atom = False
            record_start = True
            continue
        record_start = False
        if string:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                string = False
            continue
        if byte == 34:
            charge()
            string, escaped, atom = True, False, False
        elif byte in (91, 123):
            charge()
            atom = False
        elif byte in _SPACE or byte in _BOUNDARY:
            atom = False
        elif not atom:
            charge()
            atom = True
    if needed:
        _encoding_error()
    return nodes


def count_members(files, *, max_nodes=None):
    """Count .json/.jsonl bytes before a caller reconstructs or decodes them.

    File suffixes are case-sensitive. State resets between members, while the
    allowance does not. JSONL lexical state resets at every physical CR or LF,
    including malformed records. Accepted structured members have valid UTF8;
    semantic validity, balanced syntax and newline closure are not asserted.
    Zero remaining nodes is valid and refuses the first charged node.
    The default remains the per-document allowance. Callers admitting a whole
    recording may explicitly use at most MAX_TOTAL_JSON_NODES across members.
    """
    limit = io.MAX_JSON_NODES if max_nodes is None else max_nodes
    if type(limit) is not int or not 0 <= limit <= MAX_TOTAL_JSON_NODES:
        raise io.EvidenceError("raw structured-input node allowance must be a bounded nonnegative integer")
    if type(files) is not dict:
        raise io.EvidenceError("raw structured-input members must be an exact dictionary")
    nodes = 0
    for name, raw in files.items():
        if type(name) is not str or type(raw) is not bytes:
            raise io.EvidenceError("raw structured-input names and bytes have unsupported types")
        io.relative(name)
        if name.endswith((".json", ".jsonl")):
            nodes += _count_member(raw, name.endswith(".jsonl"), limit - nodes)
    return nodes
