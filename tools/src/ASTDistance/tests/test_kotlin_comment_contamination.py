#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

from pty_run import run_with_pty


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: test_kotlin_comment_contamination.py <ast_distance_bin>", file=sys.stderr)
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
/**
 * Leftover Rust KDoc must poison the file:
 * fn foo_bar(x: i32) -> i32 { x + 1 }
 */
fun fooBar(x: Int): Int {
    return x + 1
}
""".lstrip(),
            encoding="utf-8",
        )

        proc = run_with_pty(
            [
                str(ast_distance),
                "--compare-functions",
                str(rust_file),
                "rust",
                str(kotlin_file),
                "kotlin",
            ],
        )

    out = proc.stdout
    expected = [
        "*** CHEAT DETECTION FAILED ***",
        "Target file score forced to 0.0000.",
        "snake_case identifier `foo_bar` in Kotlin comments",
        "Rust `fn` declaration in Kotlin comments",
        "Function body score (missing source functions count as 0): 0.000",
    ]
    missing = [needle for needle in expected if needle not in out]
    if missing:
        raise AssertionError(
            "expected comment contamination output not found: "
            + ", ".join(missing)
            + "\n--- output ---\n"
            + out
        )


if __name__ == "__main__":
    main()
