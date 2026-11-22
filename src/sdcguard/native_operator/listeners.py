"""Bind loopback readiness to the owned server. Owner: Hari Krishna Kancharla."""
from pathlib import Path
import os
import socket

from .cleanup import processes


def require_available():
    sockets = []
    try:
        for port in (8100, 8200, 5559, 5659, 8300, 8400):
            value = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(value)
            value.bind(("127.0.0.1", port))
    except OSError as error:
        raise ValueError("a required loopback port is already occupied") from error
    finally:
        for value in sockets:
            value.close()


def _inode_owned(inodes, process, proc, rows):
    identity = getattr(process, "_owner_process_identity", {})
    anchor = rows.get(process.pid, {})
    if identity.get("pid") != process.pid or identity.get("start") != anchor.get("start"):
        return False
    if len(inodes) != 1:
        return False
    target = "socket:[" + next(iter(inodes)) + "]"
    members = [pid for pid,row in rows.items() if row.get("group") == process.pid and row.get("session") == process.pid]
    if len(members) > 128:
        raise ValueError("server process count exceeds its bound")
    for pid in members:
        try:
            paths = list((proc / str(pid) / "fd").iterdir())
            if len(paths) > 4096:
                raise ValueError("server descriptor count exceeds its bound")
            for path in paths:
                try:
                    if os.readlink(path) == target:
                        return True
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            continue
    return False


def _tcp_rows(proc):
    with (proc / "net/tcp").open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("listener inventory exceeds its bound")
    for line in raw.decode("ascii").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10:
            raise ValueError("listener inventory is malformed")
        yield fields


def owned(port, process, *, proc=Path("/proc"), table=None):
    """Require the retained session's loopback listener."""
    rows = processes() if table is None else table
    address = "0100007F:" + format(port, "04X")
    inodes = set()
    for fields in _tcp_rows(proc):
        if fields[1] == address and fields[3] == "0A":
            inodes.add(fields[9])
    return _inode_owned(inodes, process, proc, rows)


def connected_owned(sock, process, *, proc=Path("/proc"), table=None):
    """Bind the accepted peer socket, so a re-bound port cannot receive a prompt."""
    local, peer = sock.getsockname(), sock.getpeername()
    if local[0] != "127.0.0.1" or peer[0] != "127.0.0.1":
        return False
    source = "0100007F:" + format(peer[1], "04X")
    destination = "0100007F:" + format(local[1], "04X")
    rows = processes() if table is None else table
    inodes = {row[9] for row in _tcp_rows(proc)
              if row[1] == source and row[2] == destination and row[3] == "01" and row[9] != "0"}
    return _inode_owned(inodes, process, proc, rows)
