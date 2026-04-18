import Cpp from "tree-sitter-cpp";
import Python from "tree-sitter-python";
import Rust from "tree-sitter-rust";
import Kotlin from "tree-sitter-kotlin";
import TypeScript from "tree-sitter-typescript";

function check(mod, name) {
    console.log(`--- ${name} ---`);
    console.log("  mod:", typeof mod);
    if (mod.default) console.log("  mod.default:", typeof mod.default);
    if (mod.language) console.log("  mod.language:", typeof mod.language);
    if (mod.typescript) console.log("  mod.typescript:", typeof mod.typescript);
}

check(Cpp, "Cpp");
check(Python, "Python");
check(Rust, "Rust");
check(Kotlin, "Kotlin");
check(TypeScript, "TypeScript");
