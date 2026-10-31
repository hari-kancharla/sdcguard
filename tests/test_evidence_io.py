"""Local packet bounds and preservation; no engine or network is involved."""
import os
from pathlib import Path

import pytest

from sdcguard import evidence_io as io


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":1e999}',
                                 b'{"x":"\\ud800"}', b'\xff', b'{'])
def test_malformed_or_ambiguous_json_is_rejected(raw):
    with pytest.raises(io.EvidenceError):
        io.parse_json(raw)


@pytest.mark.parametrize("name", ["", "/absolute", "../outside", "a/../b", "a//b", "./a",
                                  "a\\b", "url:member", "a\x00b"])
def test_packet_paths_cannot_escape_or_change_protocol(name):
    with pytest.raises(io.EvidenceError):
        io.relative(name)


def test_regular_snapshot_preserves_bytes_and_empty_directories(tmp_path):
    root = io.create_directory(tmp_path / "packet")
    io.write_file(root, "nested/data.json", b'{"a":1}\n')
    io.make_directory(root, "empty")
    before = io.snapshot(root)
    assert before == {"nested/data.json": b'{"a":1}\n'}
    assert io.names(root, directories=True) == ["empty", "nested"]
    assert io.snapshot(root) == before


def test_exclusive_output_never_replaces_a_member(tmp_path):
    io.write_file(tmp_path, "keep.txt", b"first")
    with pytest.raises(OSError):
        io.write_file(tmp_path, "keep.txt", b"second")
    assert (tmp_path / "keep.txt").read_bytes() == b"first"


def test_reading_input_or_output_through_a_link_is_refused(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "file").write_bytes(b"preserve")
    link = tmp_path / "link"
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(io.EvidenceError):
        io.snapshot(link)
    with pytest.raises((OSError, ValueError)):
        io.create_directory(link / "new")
    (original / "member-link").symlink_to(original / "file")
    with pytest.raises(io.EvidenceError):
        io.snapshot(original)


def test_fifo_is_rejected_without_waiting_for_a_writer(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(io.EvidenceError):
        io.snapshot(tmp_path)
    with pytest.raises(io.EvidenceError):
        io.read_file(tmp_path, "pipe")


def test_file_and_directory_counts_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "MAX_FILES", 2)
    for index in range(3):
        (tmp_path / str(index)).write_bytes(b"data")
    with pytest.raises(io.EvidenceError, match="file count"):
        io.names(tmp_path)
    separate = tmp_path / "dirs"
    separate.mkdir()
    for index in range(3):
        (separate / str(index)).mkdir()
    with pytest.raises(io.EvidenceError, match="directory structure"):
        io.names(separate)


def test_packet_membership_change_during_capture_is_not_accepted(tmp_path, monkeypatch):
    (tmp_path / "one").write_bytes(b"data")
    read = io.read_file

    def changed(root, name):
        data = read(root, name)
        (tmp_path / "unexpected").write_bytes(b"new")
        return data

    monkeypatch.setattr(io, "read_file", changed)
    with pytest.raises(io.EvidenceError, match="membership changed"):
        io.snapshot(tmp_path)


def test_packet_byte_count_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "MAX_TOTAL_BYTES", 3)
    (tmp_path / "data").write_bytes(b"1234")
    with pytest.raises(io.EvidenceError, match="byte count"):
        io.snapshot(tmp_path)
