import Cpp from "tree-sitter-cpp";
function dump(obj, depth = 0, prefix = "") {
    if (depth > 2) return;
    for (let k in obj) {
        try {
            console.log(`${prefix}${k}: ${typeof obj[k]}`);
            if (typeof obj[k] === 'object' && obj[k] !== null) {
                dump(obj[k], depth + 1, prefix + "  ");
            }
        } catch(e) {}
    }
}
console.log("Dumping Cpp module structure:");
dump(Cpp);
