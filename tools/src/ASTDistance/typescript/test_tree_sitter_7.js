import Cpp from "tree-sitter-cpp";
console.log("Cpp structure:", Cpp);
console.log("Cpp.language type:", typeof Cpp.language);
if (Cpp.language) {
    console.log("Cpp.language keys:", Object.keys(Cpp.language));
    for (let k in Cpp.language) {
        console.log(`  ${k}: ${typeof Cpp.language[k]}`);
    }
}
import Parser from "tree-sitter";
const p = new Parser();
try {
    p.setLanguage(Cpp);
    console.log("p.setLanguage(Cpp) SUCCESS");
} catch(e) {
    console.log("p.setLanguage(Cpp) FAIL:", e.message);
}
