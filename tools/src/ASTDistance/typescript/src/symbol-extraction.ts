// port-lint: source include/symbol_extraction.hpp
import Parser from "tree-sitter";
import * as fs from "fs";
import * as path from "path";
import { Language } from "./types.js";

// External declarations for tree-sitter language functions
import Rust from "tree-sitter-rust";
import Kotlin from "tree-sitter-kotlin";

function getTreeSitterLanguage(lang: Language): any {
  switch (lang) {
    case Language.RUST: return Rust;
    case Language.KOTLIN: return Kotlin;
    default: throw new Error(`Unsupported language for symbol extraction: ${lang}`);
  }
}

export enum SymbolKind {
  FUNCTION,
  STRUCT,
  ENUM,
  TRAIT,
  IMPL_METHOD,
  CONST,
  TYPE_ALIAS,
  ENUM_VARIANT,
  FIELD,
}

export function symbolKindName(kind: SymbolKind): string {
  switch (kind) {
    case SymbolKind.FUNCTION: return "FUNCTION";
    case SymbolKind.STRUCT: return "STRUCT";
    case SymbolKind.ENUM: return "ENUM";
    case SymbolKind.TRAIT: return "TRAIT";
    case SymbolKind.IMPL_METHOD: return "IMPL_METHOD";
    case SymbolKind.CONST: return "CONST";
    case SymbolKind.TYPE_ALIAS: return "TYPE_ALIAS";
    case SymbolKind.ENUM_VARIANT: return "ENUM_VARIANT";
    case SymbolKind.FIELD: return "FIELD";
    default: return "UNKNOWN";
  }
}

export function symbolKindLabel(kind: SymbolKind): string {
  switch (kind) {
    case SymbolKind.FUNCTION: return "FUNCTIONS";
    case SymbolKind.STRUCT: return "STRUCTS";
    case SymbolKind.ENUM: return "ENUMS";
    case SymbolKind.TRAIT: return "TRAITS";
    case SymbolKind.IMPL_METHOD: return "IMPL METHODS";
    case SymbolKind.CONST: return "CONSTANTS";
    case SymbolKind.TYPE_ALIAS: return "TYPE ALIASES";
    case SymbolKind.ENUM_VARIANT: return "ENUM VARIANTS";
    case SymbolKind.FIELD: return "FIELDS";
    default: return "UNKNOWN";
  }
}

export enum Visibility {
  PUBLIC,
  PRIVATE,
  CRATE, // pub(crate) in Rust, internal in Kotlin
}

export function visibilityName(vis: Visibility): string {
  switch (vis) {
    case Visibility.PUBLIC: return "pub";
    case Visibility.PRIVATE: return "private";
    case Visibility.CRATE: return "pub(crate)";
    default: return "unknown";
  }
}

export interface Symbol {
  name: string;
  qualifiedName: string; // "Type::method" for impl methods
  kind: SymbolKind;
  visibility: Visibility;
  file: string; // Relative path
  line: number;
  parent: string; // For impl methods: the type being impl'd
  members: string[]; // For structs: fields, for enums: variants
  isTest: boolean; // Rust #[test] function or test-only type
  isStub: boolean; // Kotlin: placeholder/stub type (empty body, TODO markers)
  isExtension: boolean; // Kotlin: extension function
  receiverType: string; // Kotlin: receiver type for extension functions
}

export class SymbolTable {
  symbols: Symbol[] = [];
  byKind: Map<SymbolKind, Symbol[]> = new Map();
  byQualifiedName: Map<string, Symbol> = new Map();

  add(s: Symbol): void {
    this.symbols.push(s);
    const kindList = this.byKind.get(s.kind) || [];
    kindList.push(s);
    this.byKind.set(s.kind, kindList);

    if (s.qualifiedName) {
      this.byQualifiedName.set(s.qualifiedName, s);
    }
    // Also index by plain name for matching
    if (s.qualifiedName !== s.name) {
      this.byQualifiedName.set(s.name, s);
    }
  }

  find(name: string): Symbol | undefined {
    return this.byQualifiedName.get(name);
  }

  size(): number {
    return this.symbols.length;
  }
}

export interface SymbolMatch {
  rustSymbol: Symbol | null;
  kotlinSymbol: Symbol | null;
  confidence: number;
  matchReason: string; // "exact", "camelCase", "qualified"
}

export class SymbolParityReport {
  matches: SymbolMatch[] = [];
  missingInKotlin: Symbol[] = [];
  extraInKotlin: Symbol[] = [];
  coverage: Map<SymbolKind, [number, number]> = new Map(); // matched/total per kind (production)
  testCoverage: Map<SymbolKind, [number, number]> = new Map(); // matched/total per kind (tests)
  stubCount: number = 0; // Kotlin stubs detected

  print(verbose: boolean = true, missingOnly: boolean = false): void {
    console.log("==========================================================");
    console.log("SYMBOL PARITY: Rust → Kotlin");
    console.log("==========================================================");

    const primaryKinds = [
      SymbolKind.FUNCTION,
      SymbolKind.STRUCT,
      SymbolKind.ENUM,
      SymbolKind.TRAIT,
      SymbolKind.IMPL_METHOD,
    ];

    const secondaryKinds = [SymbolKind.CONST, SymbolKind.TYPE_ALIAS];

    const printKindDetail = (kind: SymbolKind, isTest: boolean) => {
      for (const m of this.matches) {
        if (!m.rustSymbol || m.rustSymbol.kind !== kind || m.rustSymbol.isTest !== isTest) continue;
        if (missingOnly) {
          if (!m.kotlinSymbol) {
            console.log(`  ${m.rustSymbol.name} (${visibilityName(m.rustSymbol.visibility)} in ${m.rustSymbol.file}:${m.rustSymbol.line})`);
          }
        } else if (verbose) {
          if (m.kotlinSymbol) {
            console.log(`  [MATCHED] ${m.rustSymbol.name} → ${m.kotlinSymbol.name} (${m.matchReason})`);
          } else {
            console.log(`  [MISSING] ${m.rustSymbol.name} (${visibilityName(m.rustSymbol.visibility)} in ${m.rustSymbol.file}:${m.rustSymbol.line})`);
          }
        } else {
          if (!m.kotlinSymbol) {
            console.log(`  [MISSING] ${m.rustSymbol.name} (${visibilityName(m.rustSymbol.visibility)} in ${m.rustSymbol.file}:${m.rustSymbol.line})`);
          }
        }
      }
    };

    const printKindGroup = (kinds: SymbolKind[], covMap: Map<SymbolKind, [number, number]>, isTest: boolean): [number, number] => {
      let groupMatched = 0;
      let groupTotal = 0;
      for (const kind of kinds) {
        const counts = covMap.get(kind);
        if (!counts || counts[1] === 0) continue;

        const [matched, total] = counts;
        const pct = (total > 0) ? (100 * matched / total) : 0;
        groupMatched += matched;
        groupTotal += total;

        console.log(`\n--- ${symbolKindLabel(kind)} (${matched}/${total} matched, ${pct.toFixed(1)}%) ---`);
        printKindDetail(kind, isTest);
      }
      return [groupMatched, groupTotal];
    };

    console.log("\n========== PRODUCTION CODE ==========");
    const [prodPriM, prodPriT] = printKindGroup(primaryKinds, this.coverage, false);
    const prodPriPct = prodPriT > 0 ? (100 * prodPriM / prodPriT) : 0;
    console.log(`\n  Definitions: ${prodPriM}/${prodPriT} (${prodPriPct.toFixed(1)}%)`);

    const [prodSecM, prodSecT] = printKindGroup(secondaryKinds, this.coverage, false);
    if (prodSecT > 0) {
      const prodSecPct = (100 * prodSecM / prodSecT);
      console.log(`\n  Supplementary: ${prodSecM}/${prodSecT} (${prodSecPct.toFixed(1)}%)`);
    }

    console.log("\n========== TEST CODE ==========");
    const [testPriM, testPriT] = printKindGroup(primaryKinds, this.testCoverage, true);
    const testPriPct = testPriT > 0 ? (100 * testPriM / testPriT) : 0;
    console.log(`\n  Definitions: ${testPriM}/${testPriT} (${testPriPct.toFixed(1)}%)`);

    const [testSecM, testSecT] = printKindGroup(secondaryKinds, this.testCoverage, true);
    if (testSecT > 0) {
        const testSecPct = (100 * testSecM / testSecT);
        console.log(`\n  Supplementary: ${testSecM}/${testSecT} (${testSecPct.toFixed(1)}%)`);
    }

    const missingProd = this.missingInKotlin.filter(s => !s.isTest).length;
    const missingTest = this.missingInKotlin.filter(s => s.isTest).length;

    let extraReal = 0, extraStubs = 0;
    for (const s of this.extraInKotlin) {
      if (s.isStub) extraStubs++;
      else extraReal++;
    }

    console.log("\n==========================================================");
    console.log("SUMMARY");
    console.log(`  Production definitions: ${prodPriM}/${prodPriT} (${prodPriPct.toFixed(1)}%)`);
    if (prodSecT > 0) {
        const prodSecPct2 = (100 * prodSecM / prodSecT);
        console.log(`  Supplementary symbols: ${prodSecM}/${prodSecT} (${prodSecPct2.toFixed(1)}%) [constants, type aliases]`);
    }
    console.log(`  Test definitions:      ${testPriM}/${testPriT} (${testPriPct.toFixed(1)}%) - ${missingTest} missing`);
    console.log(`  Extra (Kotlin-only):   ${extraReal} real + ${extraStubs} stubs`);
    if (this.stubCount > 0) {
        console.log(`  Kotlin stubs detected: ${this.stubCount}`);
    }
    console.log("==========================================================");
  }

  printJson(): void {
    const output = {
        matches: this.matches.map(m => ({
            rust: m.rustSymbol?.qualifiedName,
            kind: m.rustSymbol ? symbolKindName(m.rustSymbol.kind) : null,
            kotlin: m.kotlinSymbol?.qualifiedName,
            confidence: m.confidence,
            reason: m.matchReason,
            is_test: m.rustSymbol?.isTest,
            file: m.rustSymbol?.file,
            line: m.rustSymbol?.line
        })),
        missing: this.missingInKotlin.map(s => ({
            name: s.qualifiedName,
            kind: symbolKindName(s.kind),
            visibility: visibilityName(s.visibility),
            is_test: s.isTest,
            file: s.file,
            line: s.line
        })),
        extra: this.extraInKotlin.map(s => ({
            name: s.qualifiedName,
            kind: symbolKindName(s.kind),
            is_stub: s.isStub,
            file: s.file,
            line: s.line
        })),
        stub_count: this.stubCount,
        coverage: Object.fromEntries(Array.from(this.coverage.entries()).map(([k, v]) => [
            symbolKindName(k), 
            { matched: v[0], total: v[1], percent: v[1] > 0 ? (100 * v[0] / v[1]).toFixed(1) : 0 }
        ])),
        test_coverage: Object.fromEntries(Array.from(this.testCoverage.entries()).map(([k, v]) => [
            symbolKindName(k), 
            { matched: v[0], total: v[1], percent: v[1] > 0 ? (100 * v[0] / v[1]).toFixed(1) : 0 }
        ]))
    };
    console.log(JSON.stringify(output, null, 2));
  }
}

export interface SymbolParityOptions {
  json: boolean;
  verbose: boolean;
  missingOnly: boolean;
  includeStubs: boolean;
  filterKind: string;
  filterFile: string;
}

// Name conversion utilities
export function snakeToCamel(snake: string): string {
  return snake.replace(/(_\w)/g, (m) => m[1]!.toUpperCase());
}

export function rustQualifiedToKotlin(qualified: string): string {
  return qualified.replace(/::/g, ".");
}

// Internal recursive extraction helpers
function rustExtractVisibility(node: Parser.SyntaxNode): Visibility {
  for (const child of node.children) {
    if (child.type === "visibility_modifier") {
      if (child.children.length > 1) return Visibility.CRATE;
      return Visibility.PUBLIC;
    }
  }
  return Visibility.PRIVATE;
}

function rustExtractName(node: Parser.SyntaxNode, source: string, nameType: string): string {
  for (const child of node.namedChildren) {
    if (child.type === nameType) return source.substring(child.startIndex, child.endIndex);
  }
  return "";
}

function rustHasTestAttribute(node: Parser.SyntaxNode, source: string): boolean {
  let prev = node.previousNamedSibling;
  while (prev) {
    if (prev.type === "attribute_item") {
      const text = source.substring(prev.startIndex, prev.endIndex);
      if (text.includes("test") && !text.includes("cfg")) return true;
    } else {
      break;
    }
    prev = prev.previousNamedSibling;
  }
  return false;
}

function rustIsTestModule(node: Parser.SyntaxNode, source: string): boolean {
    let prev = node.previousNamedSibling;
    while (prev) {
        if (prev.type === "attribute_item") {
            const text = source.substring(prev.startIndex, prev.endIndex);
            if (text.includes("cfg") && text.includes("test")) return true;
        } else {
            break;
        }
        prev = prev.previousNamedSibling;
    }
    return false;
}

function rustExtractSymbolsRecursive(
    node: Parser.SyntaxNode,
    source: string,
    file: string,
    parentType: string,
    table: SymbolTable,
    inTestModule: boolean = false
): void {
  const type = node.type;

  if (type === "function_item") {
    const name = rustExtractName(node, source, "identifier");
    if (name) {
      const vis = rustExtractVisibility(node);
      const isTest = inTestModule || rustHasTestAttribute(node, source) || (vis === Visibility.PRIVATE && name.startsWith("test_"));
      table.add({
        name,
        qualifiedName: parentType ? `${parentType}::${name}` : name,
        kind: parentType ? SymbolKind.IMPL_METHOD : SymbolKind.FUNCTION,
        visibility: vis,
        file,
        line: node.startPosition.row + 1,
        parent: parentType,
        members: [],
        isTest,
        isStub: false,
        isExtension: false,
        receiverType: ""
      });
    }
  } else if (type === "struct_item") {
    const name = rustExtractName(node, source, "type_identifier");
    if (name) {
      const vis = rustExtractVisibility(node);
      const isTest = inTestModule || (vis === Visibility.PRIVATE && name.startsWith("Test"));
      const sym: Symbol = {
        name,
        qualifiedName: name,
        kind: SymbolKind.STRUCT,
        visibility: vis,
        file,
        line: node.startPosition.row + 1,
        parent: "",
        members: [],
        isTest,
        isStub: false,
        isExtension: false,
        receiverType: ""
      };
      // Extract fields
      for (const child of node.namedChildren) {
          if (child.type === "field_declaration_list") {
              for (const field of child.namedChildren) {
                  if (field.type === "field_declaration") {
                      const fname = rustExtractName(field, source, "field_identifier");
                      if (fname) sym.members.push(fname);
                  }
              }
          }
      }
      table.add(sym);
    }
  } else if (type === "enum_item") {
      const name = rustExtractName(node, source, "type_identifier");
      if (name) {
          const vis = rustExtractVisibility(node);
          const sym: Symbol = {
              name, qualifiedName: name, kind: SymbolKind.ENUM, visibility: vis,
              file, line: node.startPosition.row + 1, parent: "", members: [],
              isTest: inTestModule, isStub: false, isExtension: false, receiverType: ""
          };
          for (const child of node.namedChildren) {
              if (child.type === "enum_variant_list") {
                  for (const v of child.namedChildren) {
                      if (v.type === "enum_variant") {
                          const vname = rustExtractName(v, source, "identifier");
                          if (vname) sym.members.push(vname);
                      }
                  }
              }
          }
          table.add(sym);
      }
  } else if (type === "trait_item") {
      const name = rustExtractName(node, source, "type_identifier");
      if (name) {
          const vis = rustExtractVisibility(node);
          const sym: Symbol = {
              name, qualifiedName: name, kind: SymbolKind.TRAIT, visibility: vis,
              file, line: node.startPosition.row + 1, parent: "", members: [],
              isTest: inTestModule, isStub: false, isExtension: false, receiverType: ""
          };
          for (const child of node.namedChildren) {
              if (child.type === "declaration_list") {
                  for (const m of child.namedChildren) {
                      if (m.type === "function_signature_item" || m.type === "function_item") {
                          const mname = rustExtractName(m, source, "identifier");
                          if (mname) sym.members.push(mname);
                      }
                  }
              }
          }
          table.add(sym);
      }
  } else if (type === "impl_item") {
      // Complex "impl Trait for Type" or "impl Type" logic
      let hasFor = false;
      for (let i = 0; i < node.childCount; i++) {
          const child = node.child(i)!;
          if (!child.isNamed && source.substring(child.startIndex, child.endIndex) === "for") {
              hasFor = true; break;
          }
      }

      const extractTypeName = (n: Parser.SyntaxNode): string => {
          if (n.type === "type_identifier") return source.substring(n.startIndex, n.endIndex);
          if (n.type === "generic_type") {
              for (const c of n.namedChildren) if (c.type === "type_identifier") return source.substring(c.startIndex, c.endIndex);
          }
          if (n.type === "scoped_type_identifier") {
              for (let j = n.namedChildCount - 1; j >= 0; j--) {
                  const sub = n.namedChild(j)!;
                  if (sub.type === "type_identifier") return source.substring(sub.startIndex, sub.endIndex);
              }
          }
          return "";
      };

      let implType = "";
      if (hasFor) {
          let pastFor = false;
          for (let i = 0; i < node.childCount; i++) {
              const child = node.child(i)!;
              if (!child.isNamed && source.substring(child.startIndex, child.endIndex) === "for") { pastFor = true; continue; }
              if (pastFor && child.isNamed) { implType = extractTypeName(child); if (implType) break; }
          }
      } else {
          for (let i = 0; i < node.namedChildCount; i++) { implType = extractTypeName(node.namedChild(i)!); if (implType) break; }
      }

      if (implType) {
          for (const child of node.namedChildren) {
              if (child.type === "declaration_list") {
                  for (const method of child.namedChildren) {
                      rustExtractSymbolsRecursive(method, source, file, implType, table, inTestModule);
                  }
                  return;
              }
          }
      }
  } else if (type === "const_item") {
      const name = rustExtractName(node, source, "identifier");
      if (name) {
          table.add({
              name, qualifiedName: parentType ? `${parentType}::${name}` : name,
              kind: SymbolKind.CONST, visibility: rustExtractVisibility(node),
              file, line: node.startPosition.row + 1, parent: parentType, members: [],
              isTest: inTestModule, isStub: false, isExtension: false, receiverType: ""
          });
      }
  } else if (type === "type_item") {
      const name = rustExtractName(node, source, "type_identifier");
      if (name) {
          table.add({
              name, qualifiedName: name, kind: SymbolKind.TYPE_ALIAS, visibility: rustExtractVisibility(node),
              file, line: node.startPosition.row + 1, parent: "", members: [],
              isTest: inTestModule, isStub: false, isExtension: false, receiverType: ""
          });
      }
  }

  if (type !== "impl_item") {
      let childInTest = inTestModule;
      if (type === "mod_item") childInTest = childInTest || rustIsTestModule(node, source);
      for (const child of node.namedChildren) rustExtractSymbolsRecursive(child, source, file, parentType, table, childInTest);
  }
}

// Kotlin recursive extraction helpers
function kotlinExtractVisibility(node: Parser.SyntaxNode, source: string): Visibility {
    for (const child of node.namedChildren) {
        if (child.type === "modifiers") {
            for (const mod of child.namedChildren) {
                if (mod.type === "visibility_modifier") {
                    const text = source.substring(mod.startIndex, mod.endIndex);
                    if (text === "private") return Visibility.PRIVATE;
                    if (text === "internal") return Visibility.CRATE;
                    if (text === "public") return Visibility.PUBLIC;
                    if (text === "protected") return Visibility.PRIVATE;
                }
            }
        }
    }
    return Visibility.PUBLIC;
}

function kotlinExtractName(node: Parser.SyntaxNode, source: string): string {
    for (const child of node.namedChildren) {
        if (child.type === "simple_identifier" || child.type === "type_identifier") {
            return source.substring(child.startIndex, child.endIndex);
        }
    }
    return "";
}

function kotlinHasModifier(node: Parser.SyntaxNode, source: string, modifier: string): boolean {
    for (const child of node.namedChildren) {
        if (child.type === "modifiers") {
            for (const mod of child.namedChildren) {
                if (source.substring(mod.startIndex, mod.endIndex).includes(modifier)) return true;
            }
        }
    }
    return false;
}

function kotlinIsStubBody(node: Parser.SyntaxNode, source: string): boolean {
    const text = source.substring(node.startIndex, node.endIndex);
    if (!text) return true;
    const markers = ["TODO", "FIXME", "STUB", "placeholder", "not implemented", "throw NotImplementedError"];
    if (markers.some(m => text.includes(m))) return true;
    const trimmed = text.replace(/\s/g, "");
    return trimmed === "{}" || trimmed === "{};" || !trimmed;
}

function kotlinClassIsStub(node: Parser.SyntaxNode, source: string): boolean {
    for (const child of node.namedChildren) {
        if (child.type === "class_body") return kotlinIsStubBody(child, source);
    }
    return false;
}

function kotlinExtractSymbolsRecursive(
    node: Parser.SyntaxNode,
    source: string,
    file: string,
    parentType: string,
    table: SymbolTable
): void {
  const type = node.type;

  if (type === "function_declaration") {
    const name = kotlinExtractName(node, source);
    if (name) {
      const sym: Symbol = {
        name, visibility: kotlinExtractVisibility(node, source), file, line: node.startPosition.row + 1,
        parent: parentType, kind: SymbolKind.FUNCTION, qualifiedName: name, members: [],
        isTest: false, isStub: false, isExtension: false, receiverType: ""
      };
      
      let receiver = "";
      let foundName = false;
      for (const child of node.namedChildren) {
          if (child.type === "simple_identifier") { foundName = true; break; }
          if (child.type === "user_type") {
              receiver = source.substring(child.startIndex, child.endIndex).split("<")[0]!;
              break;
          }
      }

      if (receiver) {
          sym.isExtension = true; sym.receiverType = receiver; sym.kind = SymbolKind.IMPL_METHOD;
          sym.parent = receiver; sym.qualifiedName = `${receiver}.${name}`;
      } else if (parentType) {
          sym.kind = SymbolKind.IMPL_METHOD; sym.qualifiedName = `${parentType}.${name}`;
      }
      table.add(sym);
    }
  } else if (type === "class_declaration") {
      const name = kotlinExtractName(node, source);
      if (name) {
          const sym: Symbol = {
              name, qualifiedName: name, kind: SymbolKind.STRUCT, visibility: kotlinExtractVisibility(node, source),
              file, line: node.startPosition.row + 1, parent: "", members: [],
              isTest: false, isStub: kotlinClassIsStub(node, source), isExtension: false, receiverType: ""
          };
          if (kotlinHasModifier(node, source, "enum") || kotlinHasModifier(node, source, "sealed")) sym.kind = SymbolKind.ENUM;
          
          for (const child of node.namedChildren) {
              if (child.type === "class_body" || child.type === "enum_class_body") {
                  for (const member of child.namedChildren) {
                      if (member.type === "property_declaration") {
                          const mname = kotlinExtractName(member, source);
                          if (mname) sym.members.push(mname);
                      }
                      kotlinExtractSymbolsRecursive(member, source, file, name, table);
                  }
                  table.add(sym);
                  return;
              }
          }
          table.add(sym);
      }
  } else if (type === "interface_declaration") {
      const name = kotlinExtractName(node, source);
      if (name) {
          const sym: Symbol = {
              name, qualifiedName: name, kind: SymbolKind.TRAIT, visibility: kotlinExtractVisibility(node, source),
              file, line: node.startPosition.row + 1, parent: "", members: [],
              isTest: false, isStub: false, isExtension: false, receiverType: ""
          };
          for (const child of node.namedChildren) {
              if (child.type === "class_body") {
                  for (const member of child.namedChildren) {
                      if (member.type === "function_declaration") {
                          const mname = kotlinExtractName(member, source);
                          if (mname) sym.members.push(mname);
                      }
                      kotlinExtractSymbolsRecursive(member, source, file, name, table);
                  }
                  table.add(sym);
                  return;
              }
          }
          table.add(sym);
      }
  } else if (type === "object_declaration") {
      const name = kotlinExtractName(node, source);
      if (name) {
          if (kotlinHasModifier(node, source, "companion")) {
              for (const child of node.namedChildren) {
                  if (child.type === "class_body") {
                      for (const member of child.namedChildren) kotlinExtractSymbolsRecursive(member, source, file, parentType, table);
                      return;
                  }
              }
          } else {
              const sym: Symbol = {
                  name, qualifiedName: name, kind: SymbolKind.STRUCT, visibility: kotlinExtractVisibility(node, source),
                  file, line: node.startPosition.row + 1, parent: "", members: [],
                  isTest: false, isStub: false, isExtension: false, receiverType: ""
              };
              for (const child of node.namedChildren) {
                  if (child.type === "class_body") {
                      for (const member of child.namedChildren) kotlinExtractSymbolsRecursive(member, source, file, name, table);
                      table.add(sym);
                      return;
                  }
              }
              table.add(sym);
          }
      }
  } else if (type === "property_declaration" && !parentType) {
      const name = kotlinExtractName(node, source);
      if (name) table.add({
          name, qualifiedName: name, kind: SymbolKind.CONST, visibility: kotlinExtractVisibility(node, source),
          file, line: node.startPosition.row + 1, parent: "", members: [],
          isTest: false, isStub: false, isExtension: false, receiverType: ""
      });
  } else if (type === "type_alias") {
      const name = kotlinExtractName(node, source);
      if (name) table.add({
          name, qualifiedName: name, kind: SymbolKind.TYPE_ALIAS, visibility: kotlinExtractVisibility(node, source),
          file, line: node.startPosition.row + 1, parent: "", members: [],
          isTest: false, isStub: false, isExtension: false, receiverType: ""
      });
  }

  if (type !== "class_declaration" && type !== "interface_declaration" && type !== "object_declaration") {
      for (const child of node.namedChildren) kotlinExtractSymbolsRecursive(child, source, file, parentType, table);
  }
}

// Exported public functions
export function extractRustSymbols(rootDir: string): SymbolTable {
    const table = new SymbolTable();
    const parser = new Parser();
    parser.setLanguage(getTreeSitterLanguage(Language.RUST));

    const walk = (dir: string) => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                if (["vendor", "build", "target", ".git"].some(d => fullPath.includes(d))) continue;
                walk(fullPath);
            } else if (entry.isFile() && entry.name.endsWith(".rs")) {
                const content = fs.readFileSync(fullPath, "utf-8");
                const tree = parser.parse(content);
                rustExtractSymbolsRecursive(tree.rootNode, content, path.relative(rootDir, fullPath), "", table);
            }
        }
    };
    walk(rootDir);
    return table;
}

export function extractKotlinSymbols(rootDir: string): SymbolTable {
    const table = new SymbolTable();
    const parser = new Parser();
    parser.setLanguage(getTreeSitterLanguage(Language.KOTLIN));

    const walk = (dir: string) => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                if (["vendor", "build", ".git"].some(d => fullPath.includes(d))) continue;
                walk(fullPath);
            } else if (entry.isFile() && entry.name.endsWith(".kt")) {
                const content = fs.readFileSync(fullPath, "utf-8");
                const tree = parser.parse(content);
                kotlinExtractSymbolsRecursive(tree.rootNode, content, path.relative(rootDir, fullPath), "", table);
            }
        }
    };
    walk(rootDir);
    return table;
}

function generateMatchCandidates(rustSym: Symbol): { name: string, confidence: number, reason: string }[] {
    const candidates: { name: string, confidence: number, reason: string }[] = [];
    candidates.push({ name: rustSym.name, confidence: 1.0, reason: "exact" });

    if (rustSym.kind === SymbolKind.FUNCTION || rustSym.kind === SymbolKind.IMPL_METHOD) {
        const camel = snakeToCamel(rustSym.name);
        if (camel !== rustSym.name) candidates.push({ name: camel, confidence: 0.95, reason: "camelCase" });
    }

    if (rustSym.parent) {
        const q = `${rustSym.parent}.${rustSym.name}`;
        candidates.push({ name: q, confidence: 0.9, reason: "qualified" });
        const qc = `${rustSym.parent}.${snakeToCamel(rustSym.name)}`;
        if (qc !== q) candidates.push({ name: qc, confidence: 0.85, reason: "qualified+camelCase" });
        
        candidates.push({ name: `${rustSym.parent}.Companion.${rustSym.name}`, confidence: 0.8, reason: "companion" });
    }

    if (rustSym.name === "new" && rustSym.parent) {
        candidates.push({ name: `${rustSym.parent}.invoke`, confidence: 0.7, reason: "new→invoke" });
        candidates.push({ name: rustSym.parent, confidence: 0.65, reason: "new→constructor" });
    }

    return candidates;
}

export function buildParityReport(rust: SymbolTable, kotlin: SymbolTable): SymbolParityReport {
    const report = new SymbolParityReport();
    report.stubCount = kotlin.symbols.filter(s => s.isStub).length;

    const ktByName = new Map<string, Symbol>();
    const ktByQualified = new Map<string, Symbol>();
    for (const s of kotlin.symbols) {
        if (!ktByName.has(s.name) || (!s.isStub && ktByName.get(s.name)!.isStub)) ktByName.set(s.name, s);
        if (!ktByQualified.has(s.qualifiedName) || (!s.isStub && ktByQualified.get(s.qualifiedName)!.isStub)) ktByQualified.set(s.qualifiedName, s);
    }

    const matchedKotlin = new Set<Symbol>();
    for (const r of rust.symbols) {
        const candidates = generateMatchCandidates(r);
        let best: Symbol | null = null;
        let bestConf = 0, bestReason = "";

        for (const c of candidates) {
            let m = ktByQualified.get(c.name) || ktByName.get(c.name);
            if (m && c.confidence > bestConf) { best = m; bestConf = c.confidence; bestReason = c.reason; }
        }

        report.matches.push({ rustSymbol: r, kotlinSymbol: best, confidence: bestConf, matchReason: bestReason });
        const covMap = r.isTest ? report.testCoverage : report.coverage;
        const counts = covMap.get(r.kind) || [0, 0];
        counts[1]++;
        if (best) { matchedKotlin.add(best); counts[0]++; }
        else report.missingInKotlin.push(r);
        covMap.set(r.kind, counts);
    }

    for (const k of kotlin.symbols) if (!matchedKotlin.has(k)) report.extraInKotlin.push(k);
    return report;
}
