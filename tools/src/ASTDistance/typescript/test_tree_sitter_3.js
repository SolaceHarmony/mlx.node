import Python from "tree-sitter-python";
console.log("Python type:", typeof Python);
if (typeof Python === 'function') console.log("Python is a function!");
console.log("Python keys:", Object.keys(Python || {}));

import Cpp from "tree-sitter-cpp";
console.log("Cpp type:", typeof Cpp);
if (typeof Cpp === 'function') console.log("Cpp is a function!");

import TypeScript from "tree-sitter-typescript";
console.log("TypeScript type:", typeof TypeScript);
console.log("TypeScript keys:", Object.keys(TypeScript || {}));
if (TypeScript.typescript) {
    console.log("TypeScript.typescript type:", typeof TypeScript.typescript);
    if (typeof TypeScript.typescript === 'function') console.log("TypeScript.typescript is a function!");
}
