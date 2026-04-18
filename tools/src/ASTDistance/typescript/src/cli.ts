#!/usr/bin/env node
// port-lint: source src/main.cpp
import { Command } from "commander";
import * as fs from "fs";
import * as path from "path";
import chalk from "chalk";
import { ASTParser, parseFile } from "./ast-parser.js";
import { ASTSimilarity, printComparisonReport } from "./similarity.js";
import { Codebase, CodebaseComparator, SourceFileHelper } from "./codebase.js";
import { TaskManager, TaskStatus } from "./task-manager.js";
import { PortingAnalyzer, printFileStats, printTodoReport, printLintReport } from "./porting-utils.js";
import { cmdSymbols, cmdSymbolLookup } from "./symbol-analysis.js";
import { Language, NodeType, TreeNode } from "./types.js";

const program = new Command();

function parseLanguage(langStr: string): Language {
    const l = langStr.toLowerCase();
    if (l === "rust" || l === "rs") return Language.RUST;
    if (l === "kotlin" || l === "kt") return Language.KOTLIN;
    if (l === "cpp" || l === "c++") return Language.CPP;
    if (l === "python" || l === "py") return Language.PYTHON;
    if (l === "typescript" || l === "ts") return Language.TYPESCRIPT;
    throw new Error(`Unknown language: ${langStr} (use rust, kotlin, cpp, python, or typescript)`);
}

function languageName(lang: Language): string {
    switch (lang) {
        case Language.RUST: return "Rust";
        case Language.KOTLIN: return "Kotlin";
        case Language.CPP: return "C++";
        case Language.PYTHON: return "Python";
        case Language.TYPESCRIPT: return "TypeScript";
        default: return "Unknown";
    }
}

function dumpTree(node: TreeNode, indent: number = 0): void {
    const pad = "  ".repeat(indent);
    const typeName = node.nodeType === NodeType.UNKNOWN ? `UNKNOWN(${node.label})` : NodeType[node.nodeType];
    const suffix = node.isLeaf() ? " [leaf]" : "";
    const label = node.label && !node.isLeaf() ? ` (${node.label})` : "";
    console.log(`${pad}${typeName}${label}${suffix}`);
    for (const child of node.children) {
        dumpTree(child, indent + 1);
    }
}

async function cmdDeep(srcDir: string, srcLang: string, tgtDir: string, tgtLang: string) {
    const src = new Codebase(srcDir, srcLang);
    const tgt = new Codebase(tgtDir, tgtLang);

    console.log(`Scanning source codebase: ${srcDir}...`);
    src.scan();
    src.extractImports();
    src.buildDependencyGraph();
    src.extractPortingData();

    console.log(`Scanning target codebase: ${tgtDir}...`);
    tgt.scan();
    tgt.extractImports();
    tgt.buildDependencyGraph();
    tgt.extractPortingData();

    console.log("\nComputing similarities...");
    const comp = new CodebaseComparator(src, tgt);
    comp.findMatches();
    await comp.computeSimilarities();

    const ranked = comp.rankedForPorting();
    
    console.log("\n" + "=".repeat(80));
    console.log("DEEP PORTING ANALYSIS");
    console.log("=".repeat(80));
    
    console.log(`\nMatched files: ${comp.matches.length}`);
    console.log(`Unmatched source: ${comp.unmatchedSource.length}`);
    console.log(`Unmatched target: ${comp.unmatchedTarget.length}`);

    console.log("\nTop 20 Porting Priorities (by dependents and inverse similarity):");
    console.log(chalk.gray("  " + "Qualified Name".padEnd(30) + "Deps".padEnd(8) + "Sim".padEnd(10) + "Status"));
    console.log(chalk.gray("  " + "-".repeat(60)));

    for (const m of ranked.slice(0, 20)) {
        const simStr = (m.similarity * 100).toFixed(1) + "%";
        const status = m.isStub ? chalk.yellow("STUB") : (m.similarity > 0.9 ? chalk.green("OK") : chalk.cyan("PORTING"));
        console.log(`  ${m.sourceQualified.padEnd(30)}${m.sourceDependents.toString().padEnd(8)}${simStr.padEnd(10)}${status}`);
    }
}

program
    .name("ast_distance")
    .version("0.1.0")
    .description("Cross-language AST similarity and porting coordinator");

// Default action: compare two files
program
    .argument("<file1>")
    .argument("<lang1>")
    .argument("<file2>")
    .argument("<lang2>")
    .option("--agent <num>", "Agent session number")
    .action(async (file1, lang1, file2, lang2, options) => {
        const l1 = parseLanguage(lang1);
        const l2 = parseLanguage(lang2);
        const parser = new ASTParser();

        console.log(`Parsing ${languageName(l1)} file: ${file1}`);
        const tree1 = parser.parseFile(file1, l1);
        console.log(`Parsing ${languageName(l2)} file: ${file2}`);
        const tree2 = parser.parseFile(file2, l2);

        if (!tree1 || !tree2) {
            console.error("Error: Failed to parse one or both files.");
            process.exit(1);
        }

        const report = ASTSimilarity.compare(tree1, tree2);
        printComparisonReport(report);

        const ids1 = parser.extractIdentifiersFromFile(file1, l1);
        const ids2 = parser.extractIdentifiersFromFile(file2, l2);
        const contentSim = ASTSimilarity.combinedSimilarityWithContent(tree1, tree2, ids1, ids2);

        console.log(`\nContent-Aware Similarity: ${contentSim.toFixed(4)}`);
        
        if (parser.hasStubBodies(fs.readFileSync(file2, "utf-8"), l2)) {
            console.log(chalk.yellow("\n*** STUB DETECTED in target file ***"));
        }
    });

program.command("dump")
    .argument("<file>")
    .argument("<lang>")
    .action((file, lang) => {
        const l = parseLanguage(lang);
        const parser = new ASTParser();
        const tree = parser.parseFile(file, l);
        if (tree) dumpTree(tree);
    });

program.command("scan")
    .argument("<dir>")
    .argument("<lang>")
    .action((dir, lang) => {
        const codebase = new Codebase(dir, lang);
        codebase.scan();
        codebase.extractImports();
        codebase.printSummary();
    });

program.command("deep")
    .argument("<srcDir>")
    .argument("<srcLang>")
    .argument("<tgtDir>")
    .argument("<tgtLang>")
    .action((srcDir, srcLang, tgtDir, tgtLang) => {
        cmdDeep(srcDir, srcLang, tgtDir, tgtLang);
    });

program.command("init-tasks")
    .argument("<srcDir>")
    .argument("<srcLang>")
    .argument("<tgtDir>")
    .argument("<tgtLang>")
    .argument("<taskFile>")
    .action(async (srcDir, srcLang, tgtDir, tgtLang, taskFile) => {
        if (fs.existsSync(taskFile)) {
            console.error(`Error: task system detected (${taskFile}).`);
            process.exit(2);
        }

        console.log("=== Initializing Task File ===\n");
        const src = new Codebase(srcDir, srcLang);
        const tgt = new Codebase(tgtDir, tgtLang);
        src.scan(); src.extractImports(); src.buildDependencyGraph();
        tgt.scan(); tgt.extractImports();

        const comp = new CodebaseComparator(src, tgt);
        comp.findMatches();
        await comp.computeSimilarities();

        const tm = new TaskManager(taskFile);
        tm.source_root = srcDir; tm.target_root = tgtDir;
        tm.source_lang = srcLang; tm.target_lang = tgtLang;

        for (const sf of src.rankedByDependents()) {
            const task: any = {
                source_path: sf.relativePath,
                source_qualified: sf.qualifiedName,
                dependent_count: sf.dependentCount,
                status: TaskStatus.PENDING,
                dependencies: Array.from(sf.dependsOn),
                dependents: Array.from(sf.importedBy)
            };

            const match = comp.matches.find(m => m.sourcePath === sf.relativePath);
            if (match) {
                task.target_path = match.targetPath;
                task.similarity = match.similarity;
                if (match.similarity > 0.9 && !match.isStub) {
                    task.status = TaskStatus.COMPLETED;
                }
            } else {
                // Generate expected path
                let expected = sf.relativePath;
                if (tgtLang === "typescript") {
                    expected = SourceFileHelper.toKebabCase(path.basename(sf.relativePath, path.extname(sf.relativePath))) + ".ts";
                }
                task.target_path = expected;
            }
            tm.tasks.push(task);
        }

        tm.save();
        console.log(`Generated ${tm.tasks.length} tasks\nTask file: ${taskFile}`);
    });

program.command("assign")
    .argument("<taskFile>")
    .argument("<agentId>")
    .action((taskFile, agentId) => {
        const tm = new TaskManager(taskFile);
        const task = tm.assignNext(agentId);
        if (task) {
            tm.printAssignment(task, agentId);
        } else {
            console.log("No pending tasks available.");
        }
    });

program.command("complete")
    .argument("<taskFile>")
    .argument("<sourceQualified>")
    .option("--agent <num>", "Agent ID")
    .option("--override", "Force completion despite low similarity")
    .action(async (taskFile, sourceQualified, options) => {
        const tm = new TaskManager(taskFile);
        if (!tm.load()) process.exit(1);

        const task = tm.tasks.find(t => t.source_qualified === sourceQualified);
        if (!task) {
            console.error(`Error: Task not found: ${sourceQualified}`);
            process.exit(1);
        }

        // Verify similarity
        const parser = new ASTParser();
        const srcPath = path.join(tm.source_root, task.source_path);
        const tgtPath = path.join(tm.target_root, task.target_path);

        if (!fs.existsSync(tgtPath)) {
            console.error(`Error: Target file does not exist: ${tgtPath}`);
            process.exit(1);
        }

        const srcTree = parser.parseFile(srcPath, parseLanguage(tm.source_lang));
        const tgtTree = parser.parseFile(tgtPath, parseLanguage(tm.target_lang));
        
        const ids1 = parser.extractIdentifiersFromFile(srcPath, parseLanguage(tm.source_lang));
        const ids2 = parser.extractIdentifiersFromFile(tgtPath, parseLanguage(tm.target_lang));
        const sim = ASTSimilarity.combinedSimilarityWithContent(srcTree!, tgtTree!, ids1, ids2);

        if (sim < 0.85 && !options.override) {
            console.error(`Error: Cannot complete task with low similarity: ${sim.toFixed(6)}`);
            process.exit(1);
        }

        if (tm.completeTask(sourceQualified)) {
            console.log(`Marked as completed: ${sourceQualified}`);
        }
    });

program.command("release")
    .argument("<taskFile>")
    .argument("<sourceQualified>")
    .action((taskFile, sourceQualified) => {
        const tm = new TaskManager(taskFile);
        if (tm.releaseTask(sourceQualified)) {
            console.log(`Released task: ${sourceQualified}`);
        }
    });

program.parse();
