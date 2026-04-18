// port-lint: source include/imports.hpp
import * as fs from "fs";
import * as path from "path";
import { spawnSync } from "child_process";
import { Language } from "./types.js";

/**
 * Represents a package/namespace declaration.
 */
export interface PackageDecl {
  raw: string; // Original text
  path: string; // Normalized path (e.g., "ratatui.widgets.block")
  parts: string[]; // Split parts ["ratatui", "widgets", "block"]
}

export function packageLast(pkg: PackageDecl): string {
  return pkg.parts.length === 0 ? "" : pkg.parts[pkg.parts.length - 1]!;
}

export function packageParent(pkg: PackageDecl): string {
  if (pkg.parts.length <= 1) return "";
  return pkg.parts.slice(0, -1).join(".");
}

export function normalizePackageName(s: string): string {
  return s.replace(/[_-]/g, "").toLowerCase();
}

export function packageSimilarity(a: PackageDecl, b: PackageDecl): number {
  if (a.parts.length === 0 || b.parts.length === 0) return 0.0;

  let matches = 0;
  const minLen = Math.min(a.parts.length, b.parts.length);

  for (let i = 0; i < minLen; ++i) {
    const partA = normalizePackageName(a.parts[a.parts.length - 1 - i]!);
    const partB = normalizePackageName(b.parts[b.parts.length - 1 - i]!);
    if (partA === partB) {
      matches++;
    } else if (partA.includes(partB) || partB.includes(partA)) {
      matches++;
    } else {
      break;
    }
  }

  return matches / minLen;
}

/**
 * Represents an import/use statement.
 */
export interface Import {
  raw: string; // Original import text
  modulePath: string; // Normalized module path (e.g., "ratatui::style::Color")
  item: string; // Specific item if any (e.g., "Color")
  isWildcard: boolean; // true if "use foo::*" or "import foo.*"
}

export function importToFilePath(imp: Import): string {
  return imp.modulePath.replace(/::/g, "/").replace(/\./g, "/");
}

/**
 * Extract imports from source files using the C++ ast_distance binary.
 */
export class ImportExtractor {
  private binaryPath: string;

  constructor() {
    const possiblePaths = [
        path.join(process.cwd(), "tools/src/ASTDistance/build/ast_distance"),
        path.join(process.cwd(), "../tools/src/ASTDistance/build/ast_distance"),
        path.join(process.cwd(), "build/ast_distance"),
        path.join(process.cwd(), "../build/ast_distance")
    ];
    this.binaryPath = possiblePaths.find(p => fs.existsSync(p)) || "ast_distance";
  }

  private runBinary(args: string[]): string {
    const res = spawnSync(this.binaryPath, args, { encoding: "utf8", maxBuffer: 1024 * 1024 * 50 });
    if (res.error) return "";
    if (res.status !== 0) return "";
    return res.stdout;
  }

  extractFromFile(filepath: string): Import[] {
    // Binary --deps prints: DEP path
    const output = this.runBinary(["--deps", filepath, "dummy"]);
    const imports: Import[] = [];
    
    if (output) {
        const lines = output.split("\n");
        for (const line of lines) {
            if (line.startsWith("DEP ")) {
                const dep = line.substring(4).trim();
                imports.push({
                    raw: dep,
                    modulePath: dep,
                    item: "",
                    isWildcard: false
                });
            }
        }
    }
    
    return imports;
  }

  extractPackageFromFile(filepath: string): PackageDecl {
    // Fallback logic for package extraction (can be improved by binary --package)
    const dir = path.dirname(filepath);
    const parentParts = dir.split(path.sep).filter(p => p && p !== "." && p !== "src" && p !== "include");
    const stem = path.basename(filepath, path.extname(filepath));
    const parts = [...parentParts, stem];
    return {
        raw: "",
        path: parts.join("."),
        parts: parts
    };
  }
}
