import Python from "tree-sitter-python";
import Parser from "tree-sitter";

const handler = {
    get(target, prop) {
        const val = target[prop];
        console.log(`ACCESS: ${String(prop)} -> ${typeof val}`);
        return val;
    }
};

const proxy = new Proxy(Python, handler);
const p = new Parser();

console.log("Starting p.setLanguage(proxy)...");
try {
    p.setLanguage(proxy);
    console.log("SUCCESS");
} catch(e) {
    console.log("FAIL:", e.message);
}
