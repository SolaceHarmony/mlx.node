// port-lint: source include/port_lint.hpp
import * as fs from "fs";

/**
 * Port-lint support for tracking Rust -> Kotlin provenance.
 */

/**
 * Extract port-lint source annotation from a file.
 *
 * Searches the first 50 lines for:
 *   // port-lint: source <path>
 *   // port-lint: tests <path>
 *
 * Returns the source path if found (e.g., "core/src/codex.rs")
 */
export function extractSourceAnnotation(filePath: string): string | null {
  if (!fs.existsSync(filePath)) return null;

  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.split("\n").slice(0, 50);

  // Regex: // port-lint: source <path> OR // port-lint: tests <path>
  const pattern = /\/\/\s*port-lint:\s*(?:source|tests)\s+([^\s]+)/i;

  for (const line of lines) {
    const match = line.match(pattern);
    if (match) {
      return match[1]!.trim();
    }
  }

  return null;
}

/**
 * Check if a type/function definition has a port-lint suppression comment.
 */
export function hasSuppression(lines: string[], lineNum: number): boolean {
  if (lineNum <= 0 || lineNum > lines.length) return false;

  // Regex: // port-lint: ignore or ignore-duplicate
  const pattern = /\/\/\s*port-lint:\s*ignore(?:-duplicate)?/i;

  // Check current line (1-indexed)
  const idx = lineNum - 1;
  if (pattern.test(lines[idx]!)) return true;

  // Scan backwards through annotation lines
  let scanIdx = idx - 1;
  while (scanIdx >= 0) {
    let prevLine = lines[scanIdx]!.trim();

    if (pattern.test(prevLine)) return true;

    // Annotation line (@...) - continue scanning
    if (prevLine.startsWith("@")) {
      scanIdx--;
      continue;
    }

    // Empty line or comment - continue scanning
    if (!prevLine || prevLine.startsWith("//")) {
      scanIdx--;
      continue;
    }

    // Hit other code - stop scanning
    break;
  }

  return false;
}

/**
 * Read file lines into array for suppression checking.
 */
export function readFileLines(filePath: string): string[] {
  if (!fs.existsSync(filePath)) return [];
  return fs.readFileSync(filePath, "utf-8").split("\n");
}
