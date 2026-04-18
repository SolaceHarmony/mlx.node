import Python from "tree-sitter-python";
import Cpp from "tree-sitter-cpp";
import TypeScript from "tree-sitter-typescript";

function findLanguageFunc(mod, name) {
    if (typeof mod === 'function') return "SELF";
    if (mod.default && typeof mod.default === 'function') return "DEFAULT";
    for (let k in mod) {
        if (typeof mod[k] === 'function' && (k.toLowerCase().includes('lang') || k.toLowerCase().includes(name))) return k;
    }
    // Check nested
    if (mod.typescript && typeof mod.typescript === 'object') {
        for (let k in mod.typescript) {
            if (typeof mod.typescript[k] === 'function') return "typescript." + k;
        }
    }
    return "NOT FOUND";
}

console.log("Python lang func:", findLanguageFunc(Python, "python"));
console.log("Cpp lang func:", findLanguageFunc(Cpp, "cpp"));
console.log("TypeScript lang func:", findLanguageFunc(TypeScript, "typescript"));
