import Python from "tree-sitter-python";

console.log("Python keys:", Object.keys(Python));
console.log("nodeTypeInfo is array?", Array.isArray(Python.nodeTypeInfo));
if (Python.nodeTypeInfo) {
    console.log("nodeTypeInfo length:", Python.nodeTypeInfo.length);
    console.log("first element:", Python.nodeTypeInfo[0]);
}

import Cpp from "tree-sitter-cpp";
console.log("\nCpp keys:", Object.keys(Cpp));
console.log("nodeTypeInfo is array?", Array.isArray(Cpp.nodeTypeInfo));
if (Cpp.nodeTypeInfo) {
    console.log("nodeTypeInfo length:", Cpp.nodeTypeInfo.length);
    console.log("first element:", Cpp.nodeTypeInfo[0]);
}
