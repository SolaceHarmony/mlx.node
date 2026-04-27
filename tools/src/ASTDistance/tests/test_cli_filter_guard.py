#!/usr/bin/env python3
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: test_cli_filter_guard.py <ast_distance_bin>", file=sys.stderr)
        sys.exit(2)

    ast_distance = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rust_file = tmp / "sample.rs"
        kotlin_file = tmp / "Sample.kt"
        rust_file.write_text(
            """
pub fn foo_bar(x: i32) -> i32 {
    x + 1
}
""".lstrip(),
            encoding="utf-8",
        )
        kotlin_file.write_text(
            """
fun fooBar(x: Int): Int {
    return x + 1
}
""".lstrip(),
            encoding="utf-8",
        )

        command = (
            f"{shlex.quote(str(ast_distance))} "
            f"{shlex.quote(str(rust_file))} rust "
            f"{shlex.quote(str(kotlin_file))} kotlin "
            "2>&1 | grep -E 'REJECTED|Combined Score'"
        )
        proc = subprocess.run(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if "REDIRECT GUARD REJECTED" not in proc.stdout:
        raise AssertionError(
            "expected CLI filter guard to reject grep pipeline\n"
            f"returncode={proc.returncode}\n--- output ---\n{proc.stdout}"
        )
    if "Combined Score:" in proc.stdout:
        raise AssertionError(
            "comparison output leaked through CLI filter pipeline\n"
            f"returncode={proc.returncode}\n--- output ---\n{proc.stdout}"
        )


if __name__ == "__main__":
    main()
