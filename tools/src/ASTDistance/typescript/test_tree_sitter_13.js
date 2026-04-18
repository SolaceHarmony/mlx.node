import Cpp from "tree-sitter-cpp";
import Python from "tree-sitter-python";
import Parser from "tree-sitter";

const p = new Parser();

function findIt(mod, name) {
    console.log(`--- ${name} ---`);
    const queue = [{obj: mod, path: name}];
    const seen = new Set();
    
    while(queue.length > 0) {
        const {obj, path} = queue.shift();
        if (!obj || seen.has(obj)) continue;
        if (typeof obj === 'object' || typeof obj === 'function') seen.add(obj);

        try {
            p.setLanguage(obj);
            console.log(`  SUCCESS: ${path}`);
            return obj;
        } catch(e) {}

        if (typeof obj === 'object' && obj !== null) {
            for (let k in obj) {
                if (path.split('.').length < 4) {
                    queue.push({obj: obj[k], path: `${path}.${k}`});
                }
            }
        }
    }
    console.log(`  FAILED to find language for ${name}`);
    return null;
}

findIt(Cpp, "Cpp");
findIt(Python, "Python");
