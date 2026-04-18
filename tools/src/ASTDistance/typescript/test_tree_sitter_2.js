import Python from "tree-sitter-python";
console.log("Python nodeTypeInfo type:", typeof Python.nodeTypeInfo);
console.log("Python nodeTypeInfo isArray:", Array.isArray(Python.nodeTypeInfo));
if (Python.nodeTypeInfo) {
    console.log("Python nodeTypeInfo length:", Python.nodeTypeInfo.length);
    console.log("Python nodeTypeInfo[0]:", Python.nodeTypeInfo[0]);
}

import Cpp from "tree-sitter-cpp";
console.log("Cpp nodeTypeInfo length:", Cpp.nodeTypeInfo?.length);

import TypeScript from "tree-sitter-typescript";
console.log("TypeScript nodeTypeInfo length (typescript):", TypeScript.typescript?.nodeTypeInfo?.length);
