#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

from pty_run import run_with_pty


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: test_mod_rs_reexport_hints.py <ast_distance_bin>", file=sys.stderr)
        sys.exit(2)

    ast_distance = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rust_dir = tmp / "rust"
        rust_dir.mkdir()
        mod_rs = rust_dir / "mod.rs"
        (rust_dir / "foo_bar.rs").write_text(
            """
pub fn do_thing(x: i32) -> i32 {
    x + 1
}
""".lstrip(),
            encoding="utf-8",
        )
        mod_rs.write_text(
            """
mod foo_bar;
pub use foo_bar::do_thing;
""".lstrip(),
            encoding="utf-8",
        )

        kotlin_file = tmp / "FooBar.kt"
        kotlin_file.write_text(
            """
fun doThing(x: Int): Int {
    return x + 1
}
""".lstrip(),
            encoding="utf-8",
        )

        proc = run_with_pty(
            [
                str(ast_distance),
                "--compare-functions",
                str(mod_rs),
                "rust",
                str(kotlin_file),
                "kotlin",
            ],
        )

    out = proc.stdout
    expected = [
        "Function comparison failed: no source function bodies were extracted.",
        "Rust mod.rs reexports detected:",
        "do_thing -> expected doThing",
        "(actual: foo_bar::do_thing, likely source: foo_bar.rs)",
    ]
    missing = [needle for needle in expected if needle not in out]
    if missing:
        raise AssertionError(
            "expected mod.rs reexport hints not found: "
            + ", ".join(missing)
            + "\n--- output ---\n"
            + out
        )


if __name__ == "__main__":
    main()
