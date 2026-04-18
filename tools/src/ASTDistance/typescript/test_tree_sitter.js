import Python from "tree-sitter-python";
console.log("Python module type:", typeof Python);
console.log("Python keys:", Object.keys(Python || {}));
if (Python && typeof Python === 'object') {
    console.log("Python.default keys:", Object.keys(Python.default || {}));
    if (Python.language) console.log("Python.language keys:", Object.keys(Python.language || {}));
}

import Cpp from "tree-sitter-cpp";
console.log("Cpp keys:", Object.keys(Cpp || {}));

import TypeScript from "tree-sitter-typescript";
console.log("TypeScript keys:", Object.keys(TypeScript || {}));
