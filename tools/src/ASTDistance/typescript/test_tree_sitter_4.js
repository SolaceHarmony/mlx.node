import Python from "tree-sitter-python";
console.log("Python.language type:", typeof Python.language);
if (typeof Python.language === 'function') console.log("Python.language IS a function");

import Cpp from "tree-sitter-cpp";
console.log("Cpp.language type:", typeof Cpp.language);
if (typeof Cpp.language === 'function') console.log("Cpp.language IS a function");

import TypeScript from "tree-sitter-typescript";
console.log("TypeScript.typescript.language type:", typeof TypeScript.typescript?.language);
if (typeof TypeScript.typescript?.language === 'function') console.log("TypeScript.typescript.language IS a function");
