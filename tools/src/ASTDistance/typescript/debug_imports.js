import Cpp from "tree-sitter-cpp";
import Rust from "tree-sitter-rust";
import Kotlin from "tree-sitter-kotlin";
import Python from "tree-sitter-python";
import TypeScript from "tree-sitter-typescript";

console.log("Cpp:", typeof Cpp, Object.keys(Cpp || {}));
console.log("Rust:", typeof Rust, Object.keys(Rust || {}));
console.log("Kotlin:", typeof Kotlin, Object.keys(Kotlin || {}));
console.log("Python:", typeof Python, Object.keys(Python || {}));
console.log("TypeScript:", typeof TypeScript, Object.keys(TypeScript || {}));
