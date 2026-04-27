#!/usr/bin/env python3
import errno
import os
import pty
import subprocess
from pathlib import Path
from typing import Sequence


def run_with_pty(
    args: Sequence[str],
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            list(args),
            stdout=slave_fd,
            stderr=subprocess.STDOUT,
            close_fds=True,
            cwd=cwd,
        )
        os.close(slave_fd)
        slave_fd = -1

        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)

        stdout = b"".join(chunks).decode("utf-8", errors="replace")
        returncode = proc.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, list(args), output=stdout)
        return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr=None)
    finally:
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)
