// port-lint: source include/symbol_extractor.hpp
import Parser from "tree-sitter";

/**
 * Represents a class/struct/enum/interface definition found in source code.
 */
export interface SymbolDefinition {
  name: string; // Symbol name (e.g., "KeyEventRecord")
  type: string; // Type: "class", "struct", "interface", "enum", "data class", etc.
  filePath: string; // File where defined
  package: string; // Package/namespace
  lineNumber: number; // Line number where defined
  isPublic: boolean; // Whether exported/public
}

export function symbolQualifiedName(sym: SymbolDefinition): string {
  if (!sym.package) return sym.name;
  return `${sym.package}.${sym.name}`;
}

/**
 * Extracts symbol definitions (classes, structs, interfaces, enums) from source code.
 */
export class SymbolExtractor {
  /**
   * Extract all symbol definitions from a file's AST.
   */
  static extractSymbols(
    root: Parser.SyntaxNode,
    source: string,
    packageName: string,
    filePath: string
  ): SymbolDefinition[] {
    const symbols: SymbolDefinition[] = [];
    SymbolExtractor.extractSymbolsRecursive(
      root,
      source,
      packageName,
      filePath,
      symbols
    );
    return symbols;
  }

  private static extractSymbolsRecursive(
    node: Parser.SyntaxNode,
    source: string,
    packageName: string,
    filePath: string,
    symbols: SymbolDefinition[]
  ): void {
    const nodeType = node.type;
    const symbolType = SymbolExtractor.isSymbolDefinition(nodeType);

    if (symbolType) {
      const sym: SymbolDefinition = {
        type: symbolType,
        name: SymbolExtractor.extractSymbolName(node, source),
        package: packageName,
        filePath: filePath,
        isPublic: SymbolExtractor.isPublicSymbol(node, source),
        lineNumber: node.startPosition.row + 1,
      };

      if (sym.name) {
        symbols.push(sym);
      }
    }

    for (const child of node.children) {
      SymbolExtractor.extractSymbolsRecursive(
        child,
        source,
        packageName,
        filePath,
        symbols
      );
    }
  }

  private static getNodeText(node: Parser.SyntaxNode, source: string): string {
    return source.substring(node.startIndex, node.endIndex);
  }

  private static isSymbolDefinition(nodeType: string): string {
    // Rust
    if (nodeType === "struct_item") return "struct";
    if (nodeType === "enum_item") return "enum";
    if (nodeType === "trait_item") return "trait";
    if (nodeType === "type_item") return "type";

    // Kotlin
    if (nodeType === "class_declaration") return "class";
    if (nodeType === "object_declaration") return "object";
    if (nodeType === "interface_declaration") return "interface";
    if (nodeType === "type_alias") return "typealias";

    // C++
    if (nodeType === "class_specifier") return "class";
    if (nodeType === "struct_specifier") return "struct";
    if (nodeType === "enum_specifier") return "enum";

    // TypeScript
    if (nodeType === "class_declaration") return "class";
    if (nodeType === "interface_declaration") return "interface";
    if (nodeType === "enum_declaration") return "enum";
    if (nodeType === "type_alias_declaration") return "type";

    return "";
  }

  private static extractSymbolName(
    node: Parser.SyntaxNode,
    source: string
  ): string {
    for (const child of node.children) {
      const ct = child.type;
      if (
        ct === "type_identifier" ||
        ct === "identifier" ||
        ct === "simple_identifier"
      ) {
        const name = SymbolExtractor.getNodeText(child, source);
        if (name) return name;
      }
    }
    return "";
  }

  private static isPublicSymbol(node: Parser.SyntaxNode, source: string): boolean {
    for (const child of node.children) {
      const ct = child.type;
      // Rust
      if (ct === "visibility_modifier") {
        const text = SymbolExtractor.getNodeText(child, source);
        if (text.includes("pub")) return true;
      }
      // Kotlin
      if (ct === "modifiers") {
        const text = SymbolExtractor.getNodeText(child, source);
        if (text.includes("private") || text.includes("internal")) return false;
      }
    }

    // Default to true (Kotlin default public, Rust for simplicity in this tool)
    return true;
  }
}

export interface DuplicateSymbol {
  name: string;
  locations: string[];
}

export interface NamespaceMismatch {
  symbolName: string;
  sourcePackage: string;
  targetPackage: string;
  sourceFile: string;
  targetFile: string;
}

export interface MissingSymbol {
  name: string;
  sourceFile: string;
  sourcePackage: string;
}

/**
 * Compare symbol definitions across two codebases.
 */
export class SymbolComparator {
  static findDuplicates(
    symbolsByFile: Map<string, SymbolDefinition[]>
  ): DuplicateSymbol[] {
    const symbolLocations = new Map<string, string[]>();

    for (const [file, symbols] of symbolsByFile.entries()) {
      for (const sym of symbols) {
        const locations = symbolLocations.get(sym.name) || [];
        locations.push(file);
        symbolLocations.set(sym.name, locations);
      }
    }

    const duplicates: DuplicateSymbol[] = [];
    for (const [name, locations] of symbolLocations.entries()) {
      if (locations.length > 1) {
        duplicates.push({ name, locations });
      }
    }

    return duplicates;
  }

  static findNamespaceMismatches(
    sourceSymbols: Map<string, SymbolDefinition[]>,
    targetSymbols: Map<string, SymbolDefinition[]>
  ): NamespaceMismatch[] {
    const mismatches: NamespaceMismatch[] = [];

    const sourceByName = new Map<string, SymbolDefinition>();
    const targetByName = new Map<string, SymbolDefinition>();

    for (const symbols of sourceSymbols.values()) {
      for (const sym of symbols) sourceByName.set(sym.name, sym);
    }
    for (const symbols of targetSymbols.values()) {
      for (const sym of symbols) targetByName.set(sym.name, sym);
    }

    for (const [name, sourceSym] of sourceByName.entries()) {
      const targetSym = targetByName.get(name);
      if (targetSym) {
        const srcPkg = sourceSym.package.replace(/:/g, ".");
        if (srcPkg !== targetSym.package) {
          mismatches.push({
            symbolName: name,
            sourcePackage: sourceSym.package,
            targetPackage: targetSym.package,
            sourceFile: sourceSym.filePath,
            targetFile: targetSym.filePath,
          });
        }
      }
    }

    return mismatches;
  }

  static findMissingSymbols(
    sourceSymbols: Map<string, SymbolDefinition[]>,
    targetSymbols: Map<string, SymbolDefinition[]>
  ): MissingSymbol[] {
    const missing: MissingSymbol[] = [];
    const targetNames = new Set<string>();

    for (const symbols of targetSymbols.values()) {
      for (const sym of symbols) targetNames.add(sym.name);
    }

    for (const symbols of sourceSymbols.values()) {
      for (const sym of symbols) {
        if (!targetNames.has(sym.name)) {
          missing.push({
            name: sym.name,
            sourceFile: sym.filePath,
            sourcePackage: sym.package,
          });
        }
      }
    }

    return missing;
  }

  static printSymbolReport(
    duplicates: DuplicateSymbol[],
    mismatches: NamespaceMismatch[],
    missing: MissingSymbol[]
  ): void {
    console.log("\n=== Symbol Location Analysis ===\n");

    if (duplicates.length > 0) {
      console.log(`Duplicate Definitions (${duplicates.length}):`);
      for (const dup of duplicates) {
        console.log(`  ${dup.name} (defined in ${dup.locations.length} files):`);
        for (const loc of dup.locations) console.log(`    - ${loc}`);
      }
      console.log("");
    }

    if (mismatches.length > 0) {
      console.log(`Namespace Mismatches (${mismatches.length}):`);
      for (const mm of mismatches) {
        console.log(`  ${mm.symbolName}:`);
        console.log(`    Source: ${mm.sourcePackage} (${mm.sourceFile})`);
        console.log(`    Target: ${mm.targetPackage} (${mm.targetFile})`);
      }
      console.log("");
    }

    if (missing.length > 0) {
      console.log(`Missing Symbols (${missing.length}):`);
      for (const ms of missing) {
        console.log(`  ${ms.name} (${ms.sourcePackage}) - ${ms.sourceFile}`);
      }
      console.log("");
    }

    if (duplicates.length === 0 && mismatches.length === 0 && missing.length === 0) {
      console.log("No symbol location issues detected.\n");
    }
  }
}
