import Cpp from "tree-sitter-cpp";
console.log("Cpp keys:", Object.keys(Cpp));
if (Cpp.language) {
    console.log("Cpp.language type:", typeof Cpp.language);
    console.log("Cpp.language keys:", Object.keys(Cpp.language));
}

import Python from "tree-sitter-python";
console.log("Python keys:", Object.keys(Python));
if (Python.language) {
    console.log("Python.language type:", typeof Python.language);
}

import Parser from "tree-sitter";
const p = new Parser();
try {
    p.setLanguage(Cpp.language);
    console.log("p.setLanguage(Cpp.language) SUCCESS");
} catch(e) {
    console.log("p.setLanguage(Cpp.language) FAIL:", e.message);
}

try {
    p.setLanguage(Python.language);
    console.log("p.setLanguage(Python.language) SUCCESS");
} catch(e) {
    console.log("p.setLanguage(Python.language) FAIL:", e.message);
}
