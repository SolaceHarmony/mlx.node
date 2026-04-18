import Python from "tree-sitter-python";
import Parser from "tree-sitter";

const p = new Parser();

const manualPython = {
    name: "python",
    language: Python.language,
    nodeTypeInfo: Python.nodeTypeInfo
};

console.log("Trying manualPython...");
try {
    p.setLanguage(manualPython);
    console.log("SUCCESS");
} catch(e) {
    console.log("FAIL:", e.message);
}
