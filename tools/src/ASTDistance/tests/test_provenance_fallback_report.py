#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path

from pty_run import run_with_pty


def write_case(tmp: Path) -> tuple[Path, Path, Path, Path]:
    rust_root = tmp / "tmp" / "starlark" / "src"
    kotlin_root = tmp / "src" / "commonMain" / "kotlin" / "io" / "github" / "kotlinmania"
    rust_file = rust_root / "values" / "layout" / "aligned_size.rs"
    kotlin_file = kotlin_root / "starlark" / "values" / "layout" / "AlignedSize.kt"
    rust_file.parent.mkdir(parents=True)
    kotlin_file.parent.mkdir(parents=True)

    rust_file.write_text(
        """
pub struct AlignedSize {
    value: usize,
}

impl AlignedSize {
    pub fn new(value: usize) -> Self {
        Self { value }
    }

    pub fn get(&self) -> usize {
        self.value
    }
}
""".lstrip(),
        encoding="utf-8",
    )
    kotlin_file.write_text(
        """
// port-lint: source src/values/layout/alignedSize.rs
class AlignedSize(private val value: ULong) {
    fun get(): ULong {
        return value
    }
}
""".lstrip(),
        encoding="utf-8",
    )
    return rust_root, kotlin_root, rust_file, kotlin_file


def assert_proposal_file(proposal: Path) -> None:
    text = proposal.read_text(encoding="utf-8")
    expected_current = "// port-lint: source src/values/layout/alignedSize.rs"
    expected_proposed = "// port-lint: source values/layout/aligned_size.rs"
    if expected_current not in text:
        raise AssertionError(f"missing current header in {proposal}\n{text}")
    if expected_proposed not in text:
        raise AssertionError(f"missing proposed header in {proposal}\n{text}")
    if "fallback normalization" not in text:
        raise AssertionError(f"missing fallback reason in {proposal}\n{text}")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: test_provenance_fallback_report.py <ast_distance_bin>", file=sys.stderr)
        sys.exit(2)

    ast_distance = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rust_root, kotlin_root, rust_file, kotlin_file = write_case(tmp)

        direct = run_with_pty(
            [str(ast_distance), str(rust_file), "rust", str(kotlin_file), "kotlin"],
            cwd=tmp,
        )
        if "=== Provenance Header Fallback ===" not in direct.stdout:
            raise AssertionError(direct.stdout)
        if "Proposed: // port-lint: source values/layout/aligned_size.rs" not in direct.stdout:
            raise AssertionError(direct.stdout)
        assert_proposal_file(tmp / "port_lint_proposed_changes.md")

        deep = subprocess.run(
            [
                str(ast_distance),
                "--deep",
                str(rust_root),
                "rust",
                str(kotlin_root),
                "kotlin",
            ],
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if deep.returncode != 0:
            raise AssertionError(deep.stdout)
        if "Matched by exact header:" not in deep.stdout:
            raise AssertionError(deep.stdout)
        if "Matched by provenance fallback:" not in deep.stdout:
            raise AssertionError(deep.stdout)
        if "1 / 1" not in deep.stdout:
            raise AssertionError(deep.stdout)
        assert_proposal_file(tmp / "port_lint_proposed_changes.md")


if __name__ == "__main__":
    main()
