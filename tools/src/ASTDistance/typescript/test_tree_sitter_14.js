import Python from "tree-sitter-python";
import Parser from "tree-sitter";

console.log("Python keys:", Object.keys(Python));
const p = new Parser();

try {
    p.setLanguage(Python);
    console.log("p.setLanguage(Python) SUCCESS");
} catch(e) {
    console.log("p.setLanguage(Python) FAIL:", e.message);
}

if (Python.language) {
    try {
        p.setLanguage(Python.language);
        console.log("p.setLanguage(Python.language) SUCCESS");
    } catch(e) {
        console.log("p.setLanguage(Python.language) FAIL:", e.message);
    }
}
