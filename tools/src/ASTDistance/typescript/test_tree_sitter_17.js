import Python from "tree-sitter-python";
import Parser from "tree-sitter";

function deepProxy(obj, path) {
    return new Proxy(obj, {
        get(target, prop) {
            const val = target[prop];
            console.log(`ACCESS: ${path}.${String(prop)} -> ${typeof val}`);
            if (val && typeof val === 'object') {
                return deepProxy(val, `${path}.${String(prop)}`);
            }
            return val;
        }
    });
}

const proxy = deepProxy(Python, "Python");
const p = new Parser();

try {
    p.setLanguage(proxy);
} catch(e) {
    console.log("FAIL:", e.message);
}
