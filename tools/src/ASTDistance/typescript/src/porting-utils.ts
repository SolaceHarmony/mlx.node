// port-lint: source include/porting_utils.hpp
import * as fs from "fs";
import * as path from "path";

/**
 * Represents a TODO comment found in source code.
 */
export interface TodoItem {
  filePath: string;
  lineNum: number;
  tag: string; // e.g., "port", "semantics", "suspend-plugin"
  message: string;
  context: string[]; // Lines around the TODO

  // Optional: Kotlin line reference extracted from message
  ktLineStart?: number;
  ktLineEnd?: number;
}

export function printTodoItem(todo: TodoItem, verbose: boolean = true): void {
  console.log(
    `${todo.filePath}:${todo.lineNum}: TODO(${todo.tag || "untagged"}): ${todo.message}`
  );

  if (verbose && todo.context.length > 0) {
    console.log("  Context:");
    for (const line of todo.context) {
      console.log(`    ${line}`);
    }
  }
}

/**
 * Represents a lint error found in source code.
 */
export interface LintError {
  filePath: string;
  lineNum: number;
  type: string; // e.g., "unused_param", "missing_guard"
  message: string;
}

export function printLintError(err: LintError): void {
  console.log(`${err.filePath}:${err.lineNum}: ${err.type}: ${err.message}`);
}

/**
 * File statistics for porting analysis.
 */
export interface FileStats {
  path: string;
  relativePath: string;
  lineCount: number;
  codeLines: number; // Non-comment, non-blank lines
  commentLines: number;
  blankLines: number;
  isStub: boolean;
  hasHeaderGuard: boolean;
  transliteratedFrom: string; // Kotlin source path if found

  todos: TodoItem[];
  lintErrors: LintError[];
}

export function getCodeRatio(stats: FileStats, ktLines: number): number {
  if (ktLines === 0) return 0.0;
  return stats.lineCount / ktLines;
}

export function printFileStats(stats: FileStats): void {
  console.log(`File: ${stats.path}`);
  console.log(
    `  Lines: ${stats.lineCount} (code: ${stats.codeLines}, comments: ${stats.commentLines}, blank: ${stats.blankLines})`
  );
  if (stats.transliteratedFrom) {
    console.log(`  Transliterated from: ${stats.transliteratedFrom}`);
  }
  if (stats.isStub) console.log("  WARNING: Appears to be a stub");
  if (!stats.hasHeaderGuard) console.log("  WARNING: Missing header guard");
  console.log(`  TODOs: ${stats.todos.length}, Lint errors: ${stats.lintErrors.length}`);
}

/**
 * Group TODOs by tag.
 */
export function groupTodosByTag(todos: TodoItem[]): Map<string, TodoItem[]> {
  const grouped = new Map<string, TodoItem[]>();
  for (const todo of todos) {
    const tag = todo.tag || "untagged";
    if (!grouped.has(tag)) grouped.set(tag, []);
    grouped.get(tag)!.push(todo);
  }
  return grouped;
}

/**
 * Print a TODO report.
 */
export function printTodoReport(todos: TodoItem[], verbose: boolean = true): void {
  if (todos.length === 0) {
    console.log("No TODOs found.");
    return;
  }

  console.log("\n" + "=".repeat(80));
  console.log(`TODO REPORT - Found ${todos.length} TODO(s)`);
  console.log("=".repeat(80) + "\n");

  const grouped = groupTodosByTag(todos);
  console.log("Summary by tag:");
  for (const [tag, items] of grouped.entries()) {
    console.log(`  ${tag}: ${items.length}`);
  }
  console.log("");

  if (!verbose) {
    for (const todo of todos) {
      printTodoItem(todo, false);
    }
    return;
  }

  for (const todo of todos) {
    console.log("-".repeat(80));
    console.log(`FILE: ${todo.filePath}`);
    console.log(`LINE: ${todo.lineNum}`);
    console.log(`TAG:  ${todo.tag || "none"}`);
    console.log(`MSG:  ${todo.message}`);

    if (todo.ktLineStart && todo.ktLineStart > 0) {
      if (todo.ktLineEnd && todo.ktLineEnd > todo.ktLineStart) {
        console.log(`KT:   Lines ${todo.ktLineStart}-${todo.ktLineEnd}`);
      } else {
        console.log(`KT:   Line ${todo.ktLineStart}`);
      }
    }

    console.log("\nContext:");
    for (const line of todo.context) {
      console.log(`  ${line}`);
    }
    console.log("");
  }
}

/**
 * Print a lint report.
 */
export function printLintReport(errors: LintError[]): void {
  if (errors.length === 0) {
    console.log("No lint errors found.");
    return;
  }

  console.log("\n" + "=".repeat(80));
  console.log(`LINT REPORT - Found ${errors.length} error(s)`);
  console.log("=".repeat(80) + "\n");

  const grouped = new Map<string, LintError[]>();
  for (const err of errors) {
    if (!grouped.has(err.type)) grouped.set(err.type, []);
    grouped.get(err.type)!.push(err);
  }

  console.log("Summary by type:");
  for (const [type, items] of grouped.entries()) {
    console.log(`  ${type}: ${items.length}`);
  }
  console.log("");

  for (const err of errors) {
    printLintError(err);
  }
}

/**
 * Porting analysis utilities.
 */
export class PortingAnalyzer {
  // Keywords to ignore when checking for unused parameters
  static readonly IGNORED_KEYWORDS = new Set([
    "if", "while", "for", "switch", "catch", "when", "return",
    "sizeof", "alignof", "decltype", "static_assert", "constexpr", "template",
    "void", "int", "bool", "float", "double", "char", "short", "long", "unsigned",
    "auto", "const", "static", "virtual", "override", "final", "explicit",
    "inline", "noexcept", "nullptr", "true", "false", "this", "new", "delete",
    // Kotlin keywords that look like function calls
    "check", "require", "assert"
  ]);

  static isKotlinFile(filepath: string): boolean {
    return filepath.endsWith(".kt") || filepath.endsWith(".kts");
  }

  /**
   * Convert camelCase / PascalCase to snake_case.
   * e.g. "endArg" -> "end_arg", "ForwardHeapKind" -> "forward_heap_kind"
   */
  static camelToSnake(camel: string): string {
    let snake = "";
    for (let i = 0; i < camel.length; i++) {
      const char = camel[i]!;
      if (char >= "A" && char <= "Z") {
        if (i > 0) snake += "_";
        snake += char.toLowerCase();
      } else {
        snake += char;
      }
    }
    return snake;
  }

  /**
   * Find the project root from a Kotlin file path by locating src/commonMain or src/commonTest.
   * Handles both absolute and relative paths. Returns empty string if not found.
   */
  static findProjectRoot(filepath: string): string {
    let pos = filepath.indexOf("/src/commonMain/");
    if (pos === -1) {
      pos = filepath.indexOf("/src/commonTest/");
    }
    if (pos === -1) {
      if (filepath.startsWith("src/commonMain/")) return ".";
      pos = -1;
    }
    if (pos === -1) {
      if (filepath.startsWith("src/commonTest/")) return ".";
      pos = -1;
    }
    if (pos !== -1) {
      return filepath.substring(0, pos);
    }
    return "";
  }

  /**
   * Load the Rust source file for a Kotlin port file.
   */
  static loadRustSourceForPort(kotlinFilepath: string): string {
    const sourcePath = PortingAnalyzer.extractTransliteratedFrom(kotlinFilepath);
    if (!sourcePath) return "";

    const root = PortingAnalyzer.findProjectRoot(kotlinFilepath);
    if (!root) return "";

    const tmpDir = path.join(root, "tmp");
    if (!fs.existsSync(tmpDir) || !fs.statSync(tmpDir).isDirectory()) return "";

    const candidates: string[] = [
      path.join(tmpDir, sourcePath),
      path.join(tmpDir, "src", sourcePath),
    ];

    try {
      const entries = fs.readdirSync(tmpDir, { withFileTypes: true });
      for (const entry of entries) {
        if (entry.isDirectory()) {
          const dir = path.join(tmpDir, entry.name);
          candidates.push(path.join(dir, sourcePath));
          candidates.push(path.join(dir, "src", sourcePath));
        }
      }
    } catch {
      // ignore
    }

    for (const p of candidates) {
      try {
        if (fs.existsSync(p) && fs.statSync(p).isFile()) {
          return fs.readFileSync(p, "utf-8");
        }
      } catch {
        // ignore
      }
    }

    return "";
  }

  /**
   * Check if a Rust source file has a given parameter name as unused (prefixed with _).
   * Converts Kotlin camelCase to Rust snake_case.
   */
  static rustHasUnusedParam(rustContent: string, kotlinParam: string): boolean {
    if (!rustContent) return false;

    let bare = kotlinParam;
    if (bare.startsWith("_")) {
      bare = bare.substring(1);
    }

    const snake = PortingAnalyzer.camelToSnake(bare);
    const pattern = "_" + snake;

    try {
      const rustRe = new RegExp(`\\b${pattern}\\b`);
      return rustRe.test(rustContent);
    } catch {
      return false;
    }
  }

  /**
   * Scan a file for TODO comments.
   */
  static scanTodos(filepath: string, contextLines: number = 3): TodoItem[] {
    const todos: TodoItem[] = [];
    let content: string;
    try {
      content = fs.readFileSync(filepath, "utf-8");
    } catch {
      return todos;
    }

    const lines = content.split("\n");
    const todoRe = /\/\/\s*TODO(\([^)]*\))?:\s*(.+)/;
    const lineRefRe = /Line\s+(\d+)(?:-(\d+))?/i;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!;
      const match = line.match(todoRe);
      if (match) {
        const todo: TodoItem = {
          filePath: filepath,
          lineNum: i + 1,
          tag: "",
          message: match[2]!,
          context: [],
        };

        const tagPart = match[1];
        if (tagPart && tagPart.length > 2) {
          todo.tag = tagPart.substring(1, tagPart.length - 1); // Remove ()
        }

        const lineMatch = todo.message.match(lineRefRe);
        if (lineMatch) {
          todo.ktLineStart = parseInt(lineMatch[1]!, 10);
          if (lineMatch[2]) {
            todo.ktLineEnd = parseInt(lineMatch[2]!, 10);
          } else {
            todo.ktLineEnd = todo.ktLineStart;
          }
        }

        const start = Math.max(0, i - contextLines);
        const end = Math.min(lines.length, i + contextLines + 1);
        for (let j = start; j < end; j++) {
          const prefix = j === i ? ">>> " : "    ";
          const lineNumStr = (j + 1).toString().padStart(4, " ");
          todo.context.push(`${prefix}${lineNumStr}: ${lines[j]}`);
        }

        todos.push(todo);
      }
    }

    return todos;
  }

  /**
   * Extract source path header from a file.
   */
  static extractTransliteratedFrom(filepath: string): string {
    let content: string;
    try {
      content = fs.readFileSync(filepath, "utf-8");
    } catch {
      return "";
    }

    const transRe = /Transliterated from:\s*(.+)/i;
    const portlintRe = /port-lint:\s*(?:source|tests)\s+([^\s]+)/i;
    
    const lines = content.split("\n").slice(0, 50);
    for (const line of lines) {
      let match = line.match(transRe);
      if (match) {
        return match[1]!.trim();
      }
      match = line.match(portlintRe);
      if (match) {
        let result = match[1]!.trim();
        if (result.startsWith("codex-rs/")) {
          result = result.substring(9);
        }
        return result;
      }
    }

    return "";
  }

  /**
   * Analyze file statistics (line counts, stub detection, header guards).
   */
  static analyzeFile(filepath: string): FileStats {
    let content: string;
    try {
      content = fs.readFileSync(filepath, "utf-8");
    } catch {
      return {
        path: filepath,
        relativePath: path.basename(filepath),
        lineCount: 0,
        codeLines: 0,
        commentLines: 0,
        blankLines: 0,
        isStub: false,
        hasHeaderGuard: true,
        transliteratedFrom: "",
        todos: [],
        lintErrors: [],
      };
    }

    const lines = content.split("\n");
    const stats: FileStats = {
      path: filepath,
      relativePath: path.basename(filepath),
      lineCount: lines.length,
      codeLines: 0,
      commentLines: 0,
      blankLines: 0,
      isStub: false,
      hasHeaderGuard: true,
      transliteratedFrom: PortingAnalyzer.extractTransliteratedFrom(filepath),
      todos: PortingAnalyzer.scanTodos(filepath),
      lintErrors: PortingAnalyzer.lintFile(filepath),
    };

    let inBlockComment = false;
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.length === 0) {
        stats.blankLines++;
      } else if (trimmed.startsWith("//")) {
        stats.commentLines++;
      } else if (trimmed.includes("/*")) {
        stats.commentLines++;
        if (!trimmed.includes("*/")) {
          inBlockComment = true;
        }
      } else if (inBlockComment) {
        stats.commentLines++;
        if (trimmed.includes("*/")) {
          inBlockComment = false;
        }
      } else {
        stats.codeLines++;
      }
    }

    if (filepath.endsWith(".hpp") || filepath.endsWith(".h")) {
      stats.hasHeaderGuard = content.includes("#pragma once") || content.includes("#ifndef");
    }

    // Stub detection
    let clean = content
      .replace(/\/\/[^\n]*/g, "")
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/#include[^\n]*/g, "")
      .replace(/namespace[^\{]*\{?/g, "")
      .replace(/#pragma[^\n]*/g, "")
      .replace(/package\s+[^\n]*/g, "")
      .replace(/import\s+[^\n]*/g, "")
      .replace(/use\s+[^\n]*/g, "")
      .replace(/mod\s+\w+\s*;/g, "")
      .replace(/from\s+[^\n]*/g, "")
      .replace(/Copyright[^\n]*/g, "")
      .replace(/Licensed under[^\n]*/g, "")
      .replace(/Apache License[^\n]*/g, "")
      .replace(/\s+/g, "");

    let hasRealDeclarations = false;
    const ext = path.extname(filepath);
    if (ext === ".kt" || ext === ".kts") {
      let objectHits = 0;
      let pos = clean.indexOf("object");
      while (pos !== -1) {
        objectHits++;
        pos = clean.indexOf("object", pos + 6);
      }

      if (
        objectHits >= 2 ||
        clean.includes("fun") ||
        clean.includes("class") ||
        clean.includes("interface") ||
        clean.includes("enum") ||
        clean.includes("typealias") ||
        clean.includes("val") ||
        clean.includes("var")
      ) {
        hasRealDeclarations = true;
      }
    }

    stats.isStub = clean.length < 100 && !hasRealDeclarations;

    return stats;
  }

  /**
   * Extract parameter names from a Kotlin function parameter list.
   */
  static extractKotlinParamNames(argsStr: string): string[] {
    const params: string[] = [];
    const segments: string[] = [];
    let angleDepth = 0;
    let current = "";

    for (let i = 0; i < argsStr.length; i++) {
      const c = argsStr[i]!;
      if (c === "<") angleDepth++;
      else if (c === ">") angleDepth--;
      
      if (c === "," && angleDepth === 0) {
        segments.push(current);
        current = "";
      } else {
        current += c;
      }
    }
    if (current) segments.push(current);

    const kotlinParamRe = /\b(\w+)\s*:/;
    for (const seg of segments) {
      let s = seg;
      let adepth = 0;
      let eqPos = -1;
      for (let i = 0; i < s.length; i++) {
        if (s[i] === "<") adepth++;
        else if (s[i] === ">") adepth--;
        else if (s[i] === "=" && adepth === 0) {
          eqPos = i;
          break;
        }
      }
      if (eqPos !== -1) {
        s = s.substring(0, eqPos);
      }

      const matches = [...s.matchAll(new RegExp(kotlinParamRe, "g"))];
      const lastName = matches.length > 0 ? matches[matches.length - 1]![1] : "";

      if (lastName && !PortingAnalyzer.IGNORED_KEYWORDS.has(lastName)) {
        params.push(lastName);
      }
    }

    return params;
  }

  /**
   * Extract parameter names from a C/C++ function parameter list.
   */
  static extractCppParamNames(argsStr: string): string[] {
    const params: string[] = [];
    const segments = argsStr.split(",");

    for (let seg of segments) {
      const eqPos = seg.indexOf("=");
      if (eqPos !== -1) {
        seg = seg.substring(0, eqPos);
      }

      const tokenRe = /\b(\w+)\b/g;
      const matches = [...seg.matchAll(tokenRe)];
      const lastToken = matches.length > 0 ? matches[matches.length - 1]![1] : "";

      if (
        lastToken &&
        !PortingAnalyzer.IGNORED_KEYWORDS.has(lastToken) &&
        !lastToken.startsWith("_")
      ) {
        params.push(lastToken);
      }
    }

    return params;
  }

  static lintFile(filepath: string): LintError[] {
    const errors: LintError[] = [];
    let content: string;
    try {
      content = fs.readFileSync(filepath, "utf-8");
    } catch {
      return errors;
    }

    const kotlin = PortingAnalyzer.isKotlinFile(filepath);
    const rustContent = kotlin ? PortingAnalyzer.loadRustSourceForPort(filepath) : "";

    const funcRe = kotlin
      ? /\bfun\s+(?:<[^>]*>\s+)?(\w+)\s*\(([^)]*)\)\s*(?::\s*\w+(?:<[^>]*>)?\s*)?\{/g
      : /(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?\{/g;

    let match;
    while ((match = funcRe.exec(content)) !== null) {
      const funcName = match[1]!;
      const argsStr = match[2]!;

      if (PortingAnalyzer.IGNORED_KEYWORDS.has(funcName)) continue;

      let depth = 1;
      let idx = match.index + match[0]!.length;
      while (idx < content.length && depth > 0) {
        if (content[idx] === "{") depth++;
        else if (content[idx] === "}") depth--;
        idx++;
      }

      if (depth !== 0) continue;

      const body = content.substring(match.index + match[0]!.length, idx - 1);

      if (!argsStr || argsStr.trim() === "void") continue;

      const params = kotlin
        ? PortingAnalyzer.extractKotlinParamNames(argsStr)
        : PortingAnalyzer.extractCppParamNames(argsStr);

      for (const p of params) {
        const usageRe = new RegExp(`\\b${p}\\b`);
        if (!usageRe.test(body)) {
          if (kotlin && p.startsWith("_")) {
            if (PortingAnalyzer.rustHasUnusedParam(rustContent, p)) {
              continue;
            }
          }

          if (!kotlin) {
            if (body.includes(`(void)${p}`) || body.includes(`(void) ${p}`)) {
              continue;
            }
          }

          const lineNum = content.substring(0, match.index).split("\n").length;
          errors.push({
            filePath: filepath,
            lineNum,
            type: "unused_param",
            message: `Unused parameter '${p}' in function '${funcName}'`,
          });
        }
      }
    }

    if (filepath.endsWith(".hpp") || filepath.endsWith(".h")) {
      if (!content.includes("#pragma once") && !content.includes("#ifndef")) {
        errors.push({
          filePath: filepath,
          lineNum: 1,
          type: "missing_guard",
          message: "Missing header guard (#pragma once or #ifndef)",
        });
      }
    }

    return errors;
  }

  /**
   * Scan a directory for source files and analyze them.
   */
  static analyzeDirectory(directory: string): FileStats[] {
    const results: FileStats[] = [];

    if (!fs.existsSync(directory)) return results;

    const stat = fs.statSync(directory);
    if (stat.isFile()) {
      const ext = path.extname(directory);
      if ([".hpp", ".cpp", ".h", ".kt", ".kts", ".rs"].includes(ext)) {
        const stats = PortingAnalyzer.analyzeFile(directory);
        results.push(stats);
      }
      return results;
    }

    const walk = (dir: string) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          if (
            ["vendor", "build", "tmp", ".git", "node_modules"].some((d) =>
              fullPath.includes(`/${d}/`) || fullPath.endsWith(`/${d}`)
            )
          ) {
            continue;
          }
          walk(fullPath);
        } else if (entry.isFile()) {
          const ext = path.extname(fullPath);
          if ([".hpp", ".cpp", ".h", ".kt", ".kts", ".rs"].includes(ext)) {
            results.push(PortingAnalyzer.analyzeFile(fullPath));
          }
        }
      }
    };

    walk(directory);
    return results;
  }
}
