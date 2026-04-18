import Cpp from "tree-sitter-cpp";
import Python from "tree-sitter-python";

function inspect(mod, name) {
    console.log(`--- ${name} ---`);
    console.log("mod type:", typeof mod);
    if (mod.default) console.log("mod.default type:", typeof mod.default);
    if (mod.language) console.log("mod.language type:", typeof mod.language);
    
    const target = mod.language || mod.default?.language || mod;
    console.log("target type:", typeof target);
    if (target) {
        console.log("target keys:", Object.keys(target));
        console.log("target string:", target.toString());
        if (target.constructor) console.log("target constructor:", target.constructor.name);
    }
}

inspect(Cpp, "Cpp");
inspect(Python, "Python");
