import re
import sys
from pathlib import Path

from pty_run import run_with_pty


def parse_first_kotlin_comment_stats(output: str) -> tuple[int, int]:
    """
    Return (doc_comments, block_comments) from the first '=== Kotlin Comments ===' section.
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "=== Kotlin Comments ===":
            section = "\n".join(lines[i : i + 40])
            doc_m = re.search(r"Doc comments:\s+(\d+)", section)
            block_m = re.search(r"Block comments:\s+(\d+)", section)
            if not doc_m or not block_m:
                raise AssertionError("Failed to parse Kotlin comment stats section")
            return int(doc_m.group(1)), int(block_m.group(1))
    raise AssertionError("No Kotlin Comments section found")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: test_doc_comment_parsing.py <ast_distance_bin> <fixture_kt>", file=sys.stderr)
        return 2

    ast_distance = Path(sys.argv[1])
    fixture = Path(sys.argv[2])

    proc = run_with_pty(
        [str(ast_distance), str(fixture), "kotlin", str(fixture), "kotlin"],
    )
    out = proc.stdout

    doc_comments, block_comments = parse_first_kotlin_comment_stats(out)

    # Expected:
    # - KDoc block counts as doc (1)
    # - '///' counts as doc (1)
    # - '//!' counts as doc (1)
    # - Plain /* ... */ blocks are NOT doc (2 block comments)
    if doc_comments != 3:
        raise AssertionError(f"expected doc_comments=3, got {doc_comments}\n--- output ---\n{out}")
    if block_comments != 2:
        raise AssertionError(f"expected block_comments=2, got {block_comments}\n--- output ---\n{out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
