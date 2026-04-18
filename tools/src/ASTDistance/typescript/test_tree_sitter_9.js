import Cpp from "tree-sitter-cpp";
import Parser from "tree-sitter";

const p = new Parser();

function tryAll(obj, path = "") {
    if (!obj || typeof obj !== 'object') return;
    
    // Check if it's a function (the likely target)
    if (typeof obj === 'function') {
        try {
            p.setLanguage(obj);
            console.log(`SUCCESS at path: ${path}`);
            return true;
        } catch(e) {}
    }
    
    // Some modules also expose the pointer as a special object property
    try {
        p.setLanguage(obj);
        console.log(`SUCCESS at path: ${path} (object)`);
        return true;
    } catch(e) {}

    for (let k in obj) {
        if (path.split('.').length > 3) continue;
        if (tryAll(obj[k], path ? `${path}.${k}` : k)) return true;
    }
    return false;
}

console.log("Searching for valid language target in Cpp module...");
tryAll(Cpp, "Cpp");

import Python from "tree-sitter-python";
console.log("\nSearching for valid language target in Python module...");
tryAll(Python, "Python");
