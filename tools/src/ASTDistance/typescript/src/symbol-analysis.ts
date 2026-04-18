// port-lint: source include/symbol_analysis.hpp
import * as fs from "fs";
import * as path from "path";
import { Codebase, SourceFile } from "./codebase.js";

export interface SymbolAnalysisOptions {
  duplicates: boolean;
  stubs: boolean;
  misplaced: boolean;
  json: boolean;
  symbol: string;
}

interface CppClassDef {
  name: string;
  kind: string;
  file: string;
  line: number;
  isStub: boolean;
  stubReason: string;
}

interface StubItem {
  file: string;
  type: string;
  name: string;
  line: number;
  reason: string;
}

function shouldSkipPath(filePath: string): boolean {
  return (
    filePath.includes("/test") ||
    filePath.includes("/build/") ||
    filePath.includes("/CMakeFiles/") ||
    filePath.includes("/cmake-build") ||
    filePath.includes("/target/") ||
    filePath.includes("/_deps/")
  );
}

function removeCppComments(content: string): string {
  return content.replace(/\/\/[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "");
}

function extractClassBody(content: string, startPos: number): string {
  if (startPos >= content.length || content[startPos] !== "{") return "";
  let depth = 1;
  let idx = startPos + 1;
  while (idx < content.length && depth > 0) {
    if (content[idx] === "{") depth++;
    else if (content[idx] === "}") depth--;
    idx++;
  }
  if (depth !== 0) return "";
  if (idx <= startPos + 1) return "";
  return content.substring(startPos + 1, idx - 1);
}

function extractCppClassDefinitions(filePath: string): CppClassDef[] {
  const classes: CppClassDef[] = [];
  if (!fs.existsSync(filePath)) return classes;
  const content = fs.readFileSync(filePath, "utf-8");
  if (!content) return classes;

  const clean = removeCppComments(content);
  const classDefRe = /(?:template\s*<[^>]*>\s*)?(class|struct)\s+([A-Za-z_][\w:]*)\s*(?:\s*:\s*[^{]+)?\s*\{/gm;

  let match;
  const seen = new Set<string>();

  while ((match = classDefRe.exec(clean)) !== null) {
    const kind = match[1]!;
    const name = match[2]!;
    const line = clean.substring(0, match.index).split("\n").length;
    const key = `${name}:${line}`;
    if (seen.has(key)) continue;
    seen.add(key);

    let isStub = false;
    let stubReason = "";
    const bracePos = match.index + match[0]!.length - 1;
    const body = extractClassBody(clean, bracePos);
    if (body) {
      const trimmed = body.trim();
      if (trimmed.length < 30) {
        isStub = true;
        stubReason = "empty_body";
      } else {
        const condensed = trimmed.replace(/\s+/g, " ");
        const dtorOnlyRe = /^\s*(?:public:|private:|protected:)?\s*(?:virtual\s+)?~\w+\([^)]*\)\s*(?:=\s*default\s*)?;?\s*$/;
        if (dtorOnlyRe.test(condensed)) {
          isStub = true;
          stubReason = "only_destructor";
        }
      }
    }

    classes.push({ name, kind, file: filePath, line, isStub, stubReason });
  }

  return classes;
}

function isStubFile(filePath: string): boolean {
  if (!fs.existsSync(filePath)) return false;
  const content = fs.readFileSync(filePath, "utf-8");
  if (!content) return false;

  let clean = removeCppComments(content);
  clean = clean.replace(/#\w+[^\n]*/g, "");
  clean = clean.replace(/namespace\s+[\w:]+\s*\{/g, "");
  clean = clean.replace(/#pragma[^\n]*/g, "");
  clean = clean.replace(/\s+/g, "");

  return clean.length < 100;
}

function buildKotlinIndex(
  kotlinRoot: string,
  kotlin: Codebase
): {
  ranked: SourceFile[];
  usageCountByFile: Map<string, number>;
  usageCountByClass: Map<string, number>;
  usageCountByStem: Map<string, number>;
} {
  kotlin.scan();
  kotlin.extractImports();
  kotlin.buildDependencyGraph();
  const ranked = kotlin.rankedByDependents();

  const usageCountByFile = new Map<string, number>();
  const usageCountByClass = new Map<string, number>();
  const usageCountByStem = new Map<string, number>();

  for (const sf of kotlin.files.values()) {
    usageCountByFile.set(sf.relativePath, sf.dependentCount);
    let stem = sf.stem;
    if (stem.endsWith(".common")) stem = stem.substring(0, stem.length - 7);
    if (stem.endsWith(".native")) stem = stem.substring(0, stem.length - 7);
    usageCountByStem.set(stem, sf.dependentCount);
  }

  const classRe = /(?:class|interface|object)\s+(\w+)/g;
  for (const sf of ranked) {
    for (const p of sf.paths) {
      if (!fs.existsSync(p)) continue;
      const content = fs.readFileSync(p, "utf-8");
      let match;
      while ((match = classRe.exec(content)) !== null) {
        const name = match[1]!;
        if (!usageCountByClass.has(name)) {
          usageCountByClass.set(name, sf.dependentCount);
        }
      }
    }
  }

  return { ranked, usageCountByFile, usageCountByClass, usageCountByStem };
}

function priorityForSymbol(name: string, classDeps: Map<string, number>): number {
  return -(classDeps.get(name) || 0);
}

function priorityForFile(file: string, stemDeps: Map<string, number>): number {
  let stem = path.basename(file, path.extname(file));
  if (stem.endsWith(".common")) stem = stem.substring(0, stem.length - 7);
  if (stem.endsWith(".native")) stem = stem.substring(0, stem.length - 7);
  return -(stemDeps.get(stem) || 0);
}

export function cmdSymbols(
  kotlinRoot: string,
  cppRoot: string,
  options: SymbolAnalysisOptions
): void {
  const kotlin = new Codebase(kotlinRoot, "kotlin");
  const { usageCountByClass, usageCountByStem } = buildKotlinIndex(kotlinRoot, kotlin);

  const duplicates = new Map<string, CppClassDef[]>();
  const walk = (dir: string) => {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (shouldSkipPath(fullPath)) continue;
        walk(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(fullPath);
        if ([".hpp", ".cpp", ".h", ".cc"].includes(ext)) {
          const defs = extractCppClassDefinitions(fullPath);
          for (const cls of defs) {
            const list = duplicates.get(cls.name) || [];
            list.push(cls);
            duplicates.set(cls.name, list);
          }
        }
      }
    }
  };
  walk(cppRoot);

  const dupList: Array<{ name: string; locs: CppClassDef[] }> = [];
  for (const [name, locs] of duplicates.entries()) {
    const files = new Set(locs.map((l) => l.file));
    if (files.size > 1) {
      dupList.push({ name, locs });
    }
  }

  dupList.sort((a, b) => {
    const pa = priorityForSymbol(a.name, usageCountByClass);
    const pb = priorityForSymbol(b.name, usageCountByClass);
    if (pa !== pb) return pa - pb;
    return a.name.localeCompare(b.name);
  });

  const stubs: StubItem[] = [];
  const walkStubs = (dir: string) => {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (shouldSkipPath(fullPath)) continue;
        walkStubs(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(fullPath);
        if (ext === ".cpp" || ext === ".cc") {
          if (isStubFile(fullPath)) {
            stubs.push({
              file: path.relative(cppRoot, fullPath),
              type: "file_stub",
              name: path.basename(fullPath, ext),
              line: 0,
              reason: "",
            });
          }
        }
        if (ext === ".hpp" || ext === ".h") {
          const defs = extractCppClassDefinitions(fullPath);
          for (const cls of defs) {
            if (cls.isStub) {
              stubs.push({
                file: path.relative(cppRoot, fullPath),
                type: "class_stub",
                name: cls.name,
                line: cls.line,
                reason: cls.stubReason,
              });
            }
          }
        }
      }
    }
  };
  walkStubs(cppRoot);

  stubs.sort((a, b) => {
    let pa = priorityForSymbol(a.name, usageCountByClass);
    let pb = priorityForSymbol(b.name, usageCountByClass);
    if (pa === 0) pa = priorityForFile(a.file, usageCountByStem);
    if (pb === 0) pb = priorityForFile(b.file, usageCountByStem);
    if (pa !== pb) return pa - pb;
    return a.file.localeCompare(b.file);
  });

  console.log("======================================================================");
  console.log("SYMBOL DEFINITION ANALYSIS (ordered by dependency count)");
  console.log("======================================================================");

  if (options.duplicates || (!options.stubs && !options.misplaced)) {
    console.log("\n--- DUPLICATE DEFINITIONS (real definitions in multiple files) ---");
    console.log(`Found ${dupList.length} symbols with multiple definitions:\n`);

    const shown = Math.min(30, dupList.length);
    for (let i = 0; i < shown; i++) {
      const entry = dupList[i]!;
      console.log(`  class: ${entry.name}`);
      const byFile = new Map<string, CppClassDef[]>();
      for (const loc of entry.locs) {
        const list = byFile.get(loc.file) || [];
        list.push(loc);
        byFile.set(loc.file, list);
      }
      for (const [file, locs] of byFile.entries()) {
        let isStub = false;
        const lines = locs.map((l) => {
          if (l.isStub) isStub = true;
          return l.line;
        });
        lines.sort((a, b) => a - b);
        console.log(
          `    - ${path.relative(cppRoot, file)}:${lines.join(", ")}${isStub ? " [STUB]" : ""}`
        );
      }
    }
    if (dupList.length > shown) {
      console.log(`\n  ... and ${dupList.length - shown} more`);
    }
  }

  if (options.stubs || (!options.duplicates && !options.misplaced)) {
    const fileStubs = stubs.filter((s) => s.type === "file_stub");
    const classStubs = stubs.filter((s) => s.type === "class_stub");

    console.log("\n--- STUB IMPLEMENTATIONS (ordered by dependency) ---");
    console.log(`\nStub files (${fileStubs.length}):`);
    const shownFiles = Math.min(20, fileStubs.length);
    for (let i = 0; i < shownFiles; i++) {
      console.log(`    - ${fileStubs[i]!.file}`);
    }
    if (fileStubs.length > shownFiles) {
      console.log(`    ... and ${fileStubs.length - shownFiles} more`);
    }

    console.log(`\nStub classes (${classStubs.length}):`);
    const shownClasses = Math.min(20, classStubs.length);
    for (let i = 0; i < shownClasses; i++) {
      const s = classStubs[i]!;
      console.log(`    - ${s.name} in ${s.file}${s.reason ? ` (${s.reason})` : ""}`);
    }
    if (classStubs.length > shownClasses) {
      console.log(`    ... and ${classStubs.length - shownClasses} more`);
    }
  }

  console.log("\n======================================================================");
}

export function cmdSymbolLookup(
  kotlinRoot: string,
  cppRoot: string,
  options: SymbolAnalysisOptions
): void {
  if (!options.symbol) {
    console.error("Error: --symbols-symbol requires a symbol name");
    return;
  }

  const kotlin = new Codebase(kotlinRoot, "kotlin");
  const { ranked, usageCountByClass, usageCountByStem } = buildKotlinIndex(kotlinRoot, kotlin);

  interface Loc {
    file: string;
    isDef: boolean;
    deps: number;
    isForward?: boolean;
    refs?: number;
  }
  const ktLocations: Loc[] = [];
  const cppLocations: Loc[] = [];

  const ktRe = new RegExp(`(?:class|interface|object|fun)\\s+${options.symbol}\\b`);
  const ktRefRe = new RegExp(`\\b${options.symbol}\\b`);

  for (const sf of ranked) {
    for (const p of sf.paths) {
      if (!fs.existsSync(p)) continue;
      const content = fs.readFileSync(p, "utf-8");
      if (ktRefRe.test(content)) {
        ktLocations.push({
          file: sf.relativePath,
          isDef: ktRe.test(content),
          deps: sf.dependentCount,
        });
        break; // one per SourceFile
      }
    }
  }

  const cppDefRe = new RegExp(`(?:class|struct)\\s+${options.symbol}(?:\\s*:[^\\{]+)?\\s*\\{`);
  const cppFwdRe = new RegExp(`(?:class|struct)\\s+${options.symbol}\\s*;`);
  const cppRefRe = new RegExp(`\\b${options.symbol}\\b`, "g");

  const walkCpp = (dir: string) => {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (shouldSkipPath(fullPath)) continue;
        walkCpp(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(fullPath);
        if ([".hpp", ".cpp", ".h", ".cc"].includes(ext)) {
          const content = fs.readFileSync(fullPath, "utf-8");
          if (content.includes(options.symbol)) {
            const isDef = cppDefRe.test(content);
            const isFwd = cppFwdRe.test(content) && !isDef;
            const refs = (content.match(new RegExp(`\\b${options.symbol}\\b`, "g")) || []).length;
            if (refs > 0) {
              cppLocations.push({
                file: path.relative(cppRoot, fullPath),
                isDef,
                deps: 0,
                isForward: isFwd,
                refs,
              });
            }
          }
        }
      }
    }
  };
  walkCpp(cppRoot);

  if (options.json) {
    console.log(
      JSON.stringify(
        {
          symbol: options.symbol,
          kotlin_locations: ktLocations.map((l) => ({
            file: l.file,
            is_definition: l.isDef,
            deps: l.deps,
          })),
          cpp_locations: cppLocations.map((l) => ({
            file: l.file,
            is_definition: l.isDef,
            is_forward_decl: l.isForward,
            reference_count: l.refs,
          })),
        },
        null,
        2
      )
    );
    return;
  }

  console.log(`\n=== Analysis of '${options.symbol}' ===`);
  if (usageCountByClass.has(options.symbol)) {
    console.log(`Dependency rank: ${usageCountByClass.get(options.symbol)} files depend on this`);
  }

  console.log(`\nKotlin locations (${ktLocations.length}):`);
  for (const loc of ktLocations) {
    console.log(`  ${loc.isDef ? "[DEF] " : "[ref] "}${loc.file}${loc.deps > 0 ? ` (deps: ${loc.deps})` : ""}`);
  }

  console.log(`\nC++ locations (${cppLocations.length}):`);
  for (const loc of cppLocations) {
    const marker = loc.isDef ? "[DEF] " : loc.isForward ? "[fwd] " : "[ref] ";
    console.log(`  ${marker}${loc.file}`);
  }
}
