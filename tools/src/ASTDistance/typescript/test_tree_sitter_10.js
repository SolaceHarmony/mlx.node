import Python from "tree-sitter-python";
import Parser from "tree-sitter";

const p = new Parser();
console.log("Python type:", typeof Python);
console.log("Python keys:", Object.keys(Python || {}));

try {
    p.setLanguage(Python);
    console.log("p.setLanguage(Python) SUCCESS");
} catch(e) {
    console.log("p.setLanguage(Python) FAIL:", e.message);
}

try {
    p.setLanguage(Python.language);
    console.log("p.setLanguage(Python.language) SUCCESS");
} catch(e) {
    console.log("p.setLanguage(Python.language) FAIL:", e.message);
}

if (Python.default) {
    try {
        p.setLanguage(Python.default);
        console.log("p.setLanguage(Python.default) SUCCESS");
    } catch(e) {
        console.log("p.setLanguage(Python.default) FAIL:", e.message);
    }
}
