// port-lint: source include/codebase.hpp
import * as fs from "fs";
import * as path from "path";
import Parser from "tree-sitter";
import { 
    Language, 
    NodeType, 
    TreeNode, 
    TodoItem, 
    LintError, 
    IdentifierStats,
    FunctionInfo
} from "./types.js";
import { Import, ImportExtractor, PackageDecl, packageSimilarity, normalizePackageName } from "./imports.js";
import { ASTParser, IdentifierStatsManager } from "./ast-parser.js";
import { ASTSimilarity } from "./similarity.js";
import { PortingAnalyzer, FileStats } from "./porting-utils.js";
import { SymbolDefinition, SymbolExtractor } from "./symbol-extractor.js";

/**
 * Represents a source file with its metadata.
 */
export interface SourceFile {
  paths: string[]; // All physical paths (e.g., .hpp and .cpp)
  relativePath: string; // Relative to root
  filename: string; // Representative filename (e.g., header if paired)
  stem: string; // Filename stem (logical unit name)
  qualifiedName: string; // Disambiguated name (e.g., "widgets.Block")
  extension: string; // Representative extension

  packageDecl: PackageDecl; // Package/module declaration from source
  imports: Import[]; // Imports in this file
  importedBy: Set<string>; // Files that import this one (dependents)
  dependsOn: Set<string>; // Files this imports (dependencies)

  dependentCount: number; // Number of files that depend on this
  dependencyCount: number; // Number of files this depends on

  // For comparison
  similarityScore: number;
  matchedFile: string; // Matched file in other codebase

  // Porting analysis
  transliteratedFrom: string; // "Transliterated from:" header value
  lineCount: number;
  codeLines: number;
  isStub: boolean;
  todos: TodoItem[];
  lintErrors: LintError[];
}

export class SourceFileHelper {
  // Compute qualified name from path
  static makeQualifiedName(relPath: string): string {
    const parts: string[] = [];
    const dir = path.dirname(relPath);
    if (dir !== ".") {
        const dirParts = dir.split(path.sep).filter(p => p && p !== "." && p !== "src");
        parts.push(...dirParts);
    }

    const stem = path.basename(relPath, path.extname(relPath));
    if (parts.length > 0) {
      return `${parts[parts.length - 1]}.${stem}`;
    }
    return stem;
  }

  // Normalize name for matching (snake_case <-> PascalCase)
  static normalizeName(name: string): string {
    let result = "";
    let prevLower = false;

    for (let i = 0; i < name.length; i++) {
      const c = name[i]!;
      if (c === "_") continue;

      if (c === c.toUpperCase() && c !== c.toLowerCase() && prevLower && result.length > 0) {
        result += c.toLowerCase();
      } else {
        result += c.toLowerCase();
      }
      prevLower = (c === c.toLowerCase() && c !== c.toUpperCase());
    }
    return result;
  }

  static toKebabCase(name: string): string {
    let result = "";
    for (let i = 0; i < name.length; i++) {
      const c = name[i]!;
      if (c === "_") {
        result += "-";
      } else if (c === c.toUpperCase() && c !== c.toLowerCase() && i > 0 && 
                 name[i - 1] !== "-" && name[i - 1] !== "_") {
        result += "-" + c.toLowerCase();
      } else {
        result += c.toLowerCase();
      }
    }
    return result;
  }

  static toPascalCase(name: string): string {
    const special: Record<string, string> = {
      refcell: "RefCell",
    };

    const normalized = name.replace(/_/g, "").toLowerCase();
    if (special[normalized]) return special[normalized]!;

    let result = "";
    let capitalizeNext = true;

    for (let i = 0; i < name.length; i++) {
      const c = name[i]!;
      if (c === "_") {
        capitalizeNext = true;
        continue;
      }
      if (capitalizeNext) {
        result += c.toUpperCase();
        capitalizeNext = false;
      } else {
        result += c;
      }
    }
    return result;
  }
}

/**
 * Manages a codebase - scans files, extracts imports, builds dependency graph.
 */
export class Codebase {
  rootPath: string;
  language: string; // "rust", "kotlin", "cpp", "python", "typescript"
  files: Map<string, SourceFile> = new Map(); // keyed by logical key
  byStem: Map<string, string[]> = new Map(); // stem -> list of keys
  byQualified: Map<string, string> = new Map(); // qualifiedName -> logical key

  constructor(root: string, lang: string) {
    this.rootPath = root;
    this.language = lang;
  }

  scan(): void {
    const hasValidExt = (filePath: string) => {
      const ext = path.extname(filePath);
      if (this.language === "rust") return ext === ".rs";
      if (this.language === "kotlin") return ext === ".kt" || ext === ".kts";
      if (this.language === "cpp") return [".cpp", ".hpp", ".cc", ".h", ".cxx", ".hh"].includes(ext);
      if (this.language === "python") return ext === ".py";
      if (this.language === "typescript") return ext === ".ts" || ext === ".tsx";
      return false;
    };

    if (!fs.existsSync(this.rootPath)) return;

    if (fs.statSync(this.rootPath).isFile()) {
      if (hasValidExt(this.rootPath)) {
        const filename = path.basename(this.rootPath);
        const sf: SourceFile = this.createNewSourceFile(this.rootPath, filename, filename);
        this.files.set(sf.relativePath, sf);
        this.addToIndices(sf.relativePath, sf);
      }
      return;
    }

    const rootsToScan: string[] = [this.rootPath];
    let relBase = this.rootPath;

    if (this.language === "kotlin") {
        const marker = "/src/commonMain/kotlin/";
        const pos = this.rootPath.indexOf(marker);
        if (pos !== -1) {
            const repoRoot = this.rootPath.substring(0, pos);
            const suffix = this.rootPath.substring(pos + marker.length);
            const testRoot = path.join(repoRoot, "src", "commonTest", "kotlin", suffix);
            if (fs.existsSync(testRoot) && fs.statSync(testRoot).isDirectory()) {
                rootsToScan.push(testRoot);
                relBase = repoRoot;
            }
        }
    }

    for (const scanRoot of rootsToScan) {
        this.walkRecursive(scanRoot, relBase, hasValidExt);
    }

    // Handle duplicates
    for (const [stem, keys] of this.byStem.entries()) {
        if (keys.length > 1) {
            keys.sort((a, b) => {
                const sfA = this.files.get(a)!;
                const sfB = this.files.get(b)!;
                const aIsHeader = [".hpp", ".h", ".hxx", ".hh"].includes(sfA.extension);
                const bIsHeader = [".hpp", ".h", ".hxx", ".hh"].includes(sfB.extension);
                if (aIsHeader !== bIsHeader) return aIsHeader ? -1 : 1;
                return a.length - b.length;
            });

            const seenQualified = new Set<string>();
            for (const key of keys) {
                const sf = this.files.get(key)!;
                if (seenQualified.has(sf.qualifiedName)) {
                    const p = sf.relativePath;
                    const dir = path.dirname(p);
                    const dirParts = dir.split(path.sep).filter(part => part && part !== "." && part !== "src");
                    sf.qualifiedName = [...dirParts, sf.stem].join(".");
                }
                seenQualified.add(sf.qualifiedName);
                this.byQualified.set(sf.qualifiedName, key);
            }
        }
    }
  }

  private walkRecursive(dir: string, relBase: string, hasValidExt: (p: string) => boolean): void {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (["vendor", "build", "tmp", ".git", "node_modules", "3rdparty"].some(d => fullPath.includes(`/${d}/`) || fullPath.endsWith(`/${d}`))) {
            continue;
        }
        this.walkRecursive(fullPath, relBase, hasValidExt);
      } else if (entry.isFile()) {
        if (!hasValidExt(fullPath)) continue;

        const relPath = path.relative(relBase, fullPath);
        const stem = path.basename(fullPath, path.extname(fullPath));
        const filename = path.basename(fullPath);
        const extension = path.extname(fullPath);

        let normalizedStem = stem;
        const suffixes = [".common", ".concurrent", ".native", ".common_native", ".darwin", ".apple"];
        for (const s of suffixes) {
            if (normalizedStem.endsWith(s)) {
                normalizedStem = normalizedStem.substring(0, normalizedStem.length - s.length);
                break;
            }
        }

        const directory = path.dirname(relPath);
        const logicalKey = directory === "." ? normalizedStem : path.join(directory, normalizedStem);

        const existing = this.files.get(logicalKey);
        if (existing) {
            existing.paths.push(fullPath);
            if (extension === ".hpp" || extension === ".h") {
                existing.filename = filename;
                existing.extension = extension;
                existing.relativePath = relPath;
            }
        } else {
            const sf = this.createNewSourceFile(fullPath, relPath, filename);
            this.files.set(logicalKey, sf);
            this.addToIndices(logicalKey, sf);
        }
      }
    }
  }

  private createNewSourceFile(fullPath: string, relPath: string, filename: string): SourceFile {
    return {
      paths: [fullPath],
      relativePath: relPath,
      filename,
      stem: path.basename(relPath, path.extname(relPath)),
      extension: path.extname(relPath),
      qualifiedName: SourceFileHelper.makeQualifiedName(relPath),
      packageDecl: { raw: "", path: "", parts: [] },
      imports: [],
      importedBy: new Set(),
      dependsOn: new Set(),
      dependentCount: 0,
      dependencyCount: 0,
      similarityScore: 0,
      matchedFile: "",
      transliteratedFrom: "",
      lineCount: 0,
      codeLines: 0,
      isStub: false,
      todos: [],
      lintErrors: []
    };
  }

  private addToIndices(key: string, sf: SourceFile): void {
    const keys = this.byStem.get(sf.stem) || [];
    keys.push(key);
    this.byStem.set(sf.stem, keys);
    this.byQualified.set(sf.qualifiedName, key);
  }

  extractImports(): void {
    const extractor = new ImportExtractor();
    for (const [key, sf] of this.files.entries()) {
      for (const p of sf.paths) {
        try {
            const fileImports = extractor.extractFromFile(p);
            sf.imports.push(...fileImports);

            if (this.language === "python" && sf.packageDecl.parts.length === 0) {
                sf.packageDecl = extractor.extractPackageFromFile(p);
            } else if (sf.packageDecl.parts.length === 0) {
                sf.packageDecl = extractor.extractPackageFromFile(p);
            }
        } catch (e) {
            console.error(`ERROR extracting imports/package for ${p}: ${(e as Error).message}. Stack: ${(e as Error).stack}`);
        }
      }
      sf.dependencyCount = sf.imports.length;
    }
  }

  extractPortingData(): void {
    for (const [key, sf] of this.files.entries()) {
        let allPartsStub = sf.paths.length > 0;
        for (const p of sf.paths) {
            if (!sf.transliteratedFrom) {
                sf.transliteratedFrom = PortingAnalyzer.extractTransliteratedFrom(p);
            }
            const stats = PortingAnalyzer.analyzeFile(p);
            sf.lineCount += stats.lineCount;
            sf.codeLines += stats.codeLines;
            sf.todos.push(...stats.todos);
            sf.lintErrors.push(...stats.lintErrors);
            allPartsStub = allPartsStub && stats.isStub;
        }
        sf.isStub = allPartsStub && sf.codeLines <= 100;
    }
  }

  transliterationMap(): Map<string, string> {
    const result = new Map<string, string>();
    for (const [key, sf] of this.files.entries()) {
      if (sf.transliteratedFrom) {
        result.set(sf.transliteratedFrom, key);
      }
    }
    return result;
  }

  buildDependencyGraph(): void {
    for (const [key, sf] of this.files.entries()) {
      for (const imp of sf.imports) {
        const resolved = this.resolveImport(imp);
        if (resolved && resolved !== key) {
          sf.dependsOn.add(resolved);
          this.files.get(resolved)?.importedBy.add(key);
        }
      }
    }
    for (const sf of this.files.values()) {
        sf.dependentCount = sf.importedBy.size;
    }
  }

  rankedByDependents(): SourceFile[] {
    return Array.from(this.files.values()).sort((a, b) => b.dependentCount - a.dependentCount);
  }

  leafFiles(): SourceFile[] {
    return Array.from(this.files.values()).filter(sf => sf.dependentCount === 0);
  }

  rootFiles(minDependents: number = 3): SourceFile[] {
    return Array.from(this.files.values())
      .filter(sf => sf.dependentCount >= minDependents)
      .sort((a, b) => b.dependentCount - a.dependentCount);
  }

  printSummary(): void {
    console.log(`Codebase: ${this.rootPath} (${this.language})`);
    console.log(`  Files: ${this.files.size}`);

    let totalImports = 0;
    let maxDependents = 0;
    let mostDepended = "";

    for (const sf of this.files.values()) {
      totalImports += sf.imports.length;
      if (sf.dependentCount > maxDependents) {
        maxDependents = sf.dependentCount;
        mostDepended = sf.qualifiedName;
      }
    }

    console.log(`  Total imports: ${totalImports}`);
    if (mostDepended) {
      console.log(`  Most depended: ${mostDepended} (${maxDependents} dependents)`);
    }
  }

  private resolveImport(imp: Import): string | null {
    let item = imp.item;
    if (item === "*") {
      const lastSep = imp.modulePath.lastIndexOf(this.language === "rust" ? "::" : ".");
      if (lastSep !== -1) {
        item = imp.modulePath.substring(lastSep + (this.language === "rust" ? 2 : 1));
      }
    }

    const normalized = SourceFileHelper.normalizeName(item);
    for (const [stem, keys] of this.byStem.entries()) {
      if (SourceFileHelper.normalizeName(stem) === normalized) {
        return keys[0] || null;
      }
    }
    return null;
  }
}

/**
 * Compare two codebases and find matches.
 */
export class CodebaseComparator {
  source: Codebase;
  target: Codebase;

  matches: Match[] = [];
  unmatchedSource: string[] = [];
  unmatchedTarget: string[] = [];

  constructor(src: Codebase, tgt: Codebase) {
    this.source = src;
    this.target = tgt;
  }

  static isHeaderFile(file: SourceFile): boolean {
    return [".hpp", ".h", ".hxx", ".hh"].includes(file.extension);
  }

  static nameMatchScore(src: SourceFile, tgt: SourceFile): number {
    const srcNorm = SourceFileHelper.normalizeName(src.stem);
    const tgtNorm = SourceFileHelper.normalizeName(tgt.stem);
    const srcQualNorm = SourceFileHelper.normalizeName(src.qualifiedName);
    const tgtQualNorm = SourceFileHelper.normalizeName(tgt.qualifiedName);

    const headerBoost = CodebaseComparator.isHeaderFile(tgt) ? 0.02 : 0.0;

    if (srcQualNorm === tgtQualNorm) return 1.0 + headerBoost;

    const srcDot = src.qualifiedName.lastIndexOf(".");
    const tgtDot = tgt.qualifiedName.lastIndexOf(".");
    const srcParent = srcDot !== -1 ? src.qualifiedName.substring(0, srcDot) : "";
    const tgtParent = tgtDot !== -1 ? tgt.qualifiedName.substring(0, tgtDot) : "";

    if (srcNorm === tgtNorm && srcParent && tgtParent) {
      if (SourceFileHelper.normalizeName(srcParent) === SourceFileHelper.normalizeName(tgtParent)) {
        return 0.95 + headerBoost;
      }
    }

    if (srcNorm === tgtNorm) return 0.7 + headerBoost;

    if (tgtNorm.includes(srcNorm)) return 0.5 + 0.2 * (srcNorm.length / tgtNorm.length) + headerBoost;
    if (srcNorm.includes(tgtNorm)) return 0.5 + 0.2 * (tgtNorm.length / srcNorm.length) + headerBoost;

    if (src.packageDecl.parts.length > 0 && tgt.packageDecl.parts.length > 0) {
      const pkgSim = packageSimilarity(src.packageDecl, tgt.packageDecl);
      if (pkgSim > 0.5) return pkgSim * 0.6 + headerBoost;
    }

    return 0.0;
  }

  findMatches(): void {
    const matchedSources = new Set<string>();
    const matchedTargets = new Set<string>();

    // First pass: Match by header
    const headerCandidates: Array<{score: number, srcKey: string, tgtKey: string}> = [];
    for (const [tgtKey, tgtFile] of this.target.files.entries()) {
      if (!tgtFile.transliteratedFrom) continue;
      for (const [srcKey, srcFile] of this.source.files.entries()) {
        let score = 0;
        if (tgtFile.transliteratedFrom.includes(srcFile.relativePath)) score = 1.0;
        else if (tgtFile.transliteratedFrom.endsWith("/" + srcFile.filename) || tgtFile.transliteratedFrom === srcFile.filename) {
            score = 0.9;
        } else if (tgtFile.transliteratedFrom.endsWith(srcFile.stem + ".kt") || tgtFile.transliteratedFrom.endsWith(srcFile.stem + ".rs")) {
            score = 0.3;
        }
        if (score > 0) headerCandidates.push({score, srcKey, tgtKey});
      }
    }

    headerCandidates.sort((a, b) => {
        if (Math.abs(a.score - b.score) > 0.001) return b.score - a.score;
        const tgtA = this.target.files.get(a.tgtKey)!;
        const tgtB = this.target.files.get(b.tgtKey)!;
        const aH = CodebaseComparator.isHeaderFile(tgtA);
        const bH = CodebaseComparator.isHeaderFile(tgtB);
        if (aH !== bH) return aH ? -1 : 1;
        return a.tgtKey.length - b.tgtKey.length;
    });

    for (const c of headerCandidates) {
        if (matchedSources.has(c.srcKey) || matchedTargets.has(c.tgtKey)) continue;
        const src = this.source.files.get(c.srcKey)!;
        const tgt = this.target.files.get(c.tgtKey)!;
        const m = this.createMatch(c.srcKey, c.tgtKey, src, tgt, true);
        this.matches.push(m);
        matchedSources.add(c.srcKey);
        matchedTargets.add(c.tgtKey);
    }

    // Second pass: Name match
    const nameCandidates: Array<{score: number, srcKey: string, tgtKey: string}> = [];
    for (const [srcKey, srcFile] of this.source.files.entries()) {
      if (matchedSources.has(srcKey)) continue;
      for (const [tgtKey, tgtFile] of this.target.files.entries()) {
        if (matchedTargets.has(tgtKey)) continue;
        const score = CodebaseComparator.nameMatchScore(srcFile, tgtFile);
        if (score > 0.4) nameCandidates.push({score, srcKey, tgtKey});
      }
    }

    nameCandidates.sort((a, b) => b.score - a.score);
    for (const c of nameCandidates) {
        if (matchedSources.has(c.srcKey) || matchedTargets.has(c.tgtKey)) continue;
        const src = this.source.files.get(c.srcKey)!;
        const tgt = this.target.files.get(c.tgtKey)!;
        const m = this.createMatch(c.srcKey, c.tgtKey, src, tgt, false);
        this.matches.push(m);
        matchedSources.add(c.srcKey);
        matchedTargets.add(c.tgtKey);
    }

    for (const key of this.source.files.keys()) if (!matchedSources.has(key)) this.unmatchedSource.push(key);
    for (const key of this.target.files.keys()) if (!matchedTargets.has(key)) this.unmatchedTarget.push(key);
  }

  private createMatch(srcKey: string, tgtKey: string, src: SourceFile, tgt: SourceFile, byHeader: boolean): Match {
      const m: Match = {
          sourcePath: srcKey,
          targetPath: tgtKey,
          sourceQualified: src.qualifiedName,
          targetQualified: tgt.qualifiedName,
          similarity: 0,
          sourceDependents: src.dependentCount,
          targetDependents: tgt.dependentCount,
          sourceLines: src.lineCount,
          targetLines: tgt.lineCount,
          todoCount: tgt.todos.length,
          lintCount: tgt.lintErrors.length,
          isStub: tgt.isStub,
          matchedByHeader: byHeader,
          sourceFunctionCount: 0,
          targetFunctionCount: 0,
          matchedFunctionCount: 0,
          functionCoverage: 1.0,
          sourceTypeCount: 0,
          targetTypeCount: 0,
          matchedTypeCount: 0,
          typeCoverage: 1.0,
          missingTypes: [],
          sourceDocLines: 0,
          targetDocLines: 0,
          sourceDocComments: 0,
          targetDocComments: 0,
          docSimilarity: 0,
          docCoverage: 1.0,
          docWeighted: 0
      };
      if (!m.isStub && src.codeLines > 20 && tgt.codeLines > 0) {
          if (tgt.codeLines / src.codeLines < 0.3) m.isStub = true;
      }
      return m;
  }

  async computeSimilarities(): Promise<void> {
    const parser = new ASTParser();
    const srcLangStr = this.source.language;
    const tgtLangStr = this.target.language;
    const srcLang = this.stringToLanguage(srcLangStr);
    const tgtLang = this.stringToLanguage(tgtLangStr);

    for (const m of this.matches) {
      try {
        const srcFile = this.source.files.get(m.sourcePath)!;
        const tgtFile = this.target.files.get(m.targetPath)!;

        if (parser.hasStubBodiesInFiles(tgtFile.paths, tgtLang)) {
          m.similarity = 0;
          m.isStub = true;
        } else {
          const srcTree = parser.parseFile(srcFile.paths, srcLang);
          const tgtTree = parser.parseFile(tgtFile.paths, tgtLang);
          if (srcTree) srcTree.flattenNodeType(NodeType.PACKAGE);
          if (tgtTree) tgtTree.flattenNodeType(NodeType.PACKAGE);

          if (srcTree && tgtTree) {
              const srcIds = parser.extractIdentifiersFromFile(srcFile.paths, srcLang);
              const tgtIds = parser.extractIdentifiersFromFile(tgtFile.paths, tgtLang);

              const fileSim = ASTSimilarity.combinedSimilarityWithContent(srcTree, tgtTree, srcIds, tgtIds);

              const srcFuncs = parser.extractFunctionInfosFromFiles(srcFile.paths, srcLang);
              const tgtFuncs = parser.extractFunctionInfosFromFiles(tgtFile.paths, tgtLang);
              const fnCov = this.functionNameCoverageWithLang(srcFuncs, tgtFuncs, srcLang, tgtLang);

              m.sourceFunctionCount = fnCov.sourceTotal;
              m.targetFunctionCount = fnCov.targetTotal;
              m.matchedFunctionCount = fnCov.matched;
              m.functionCoverage = fnCov.ratio;
              m.similarity = fileSim * fnCov.ratio;
          } else {
              m.similarity = 0;
          }
        }

        const srcDocs = parser.extractCommentsFromFile(srcFile.paths, srcLang);
        const tgtDocs = parser.extractCommentsFromFile(tgtFile.paths, tgtLang);
        m.sourceDocLines = srcDocs.totalDocLines;
        m.targetDocLines = tgtDocs.totalDocLines;
        m.sourceDocComments = srcDocs.docCommentCount;
        m.targetDocComments = tgtDocs.docCommentCount;
      } catch (e) {
        m.similarity = -1;
      }
    }
  }

  private stringToLanguage(lang: string): Language {
    const l = lang.toLowerCase();
    if (l === "rust" || l === "rs") return Language.RUST;
    if (l === "kotlin" || l === "kt") return Language.KOTLIN;
    if (l === "cpp" || l === "c++") return Language.CPP;
    if (l === "python" || l === "py") return Language.PYTHON;
    if (l === "typescript" || l === "ts") return Language.TYPESCRIPT;
    return Language.KOTLIN;
  }

  private functionNameCoverageWithLang(src: FunctionInfo[], tgt: FunctionInfo[], srcL: Language, tgtL: Language) {
      // Basic name-based set overlap
      const srcNames = new Set(src.filter(f => !f.isTest).map(f => IdentifierStatsManager.canonicalize(f.name)));
      const tgtNames = new Set(tgt.map(f => IdentifierStatsManager.canonicalize(f.name)));
      let matched = 0;
      for (const name of srcNames) if (tgtNames.has(name)) matched++;
      return {
          sourceTotal: srcNames.size,
          targetTotal: tgtNames.size,
          matched,
          ratio: srcNames.size > 0 ? matched / srcNames.size : 1.0
      };
  }

  rankedForPorting(): Match[] {
    return [...this.matches].sort((a, b) => {
      const scoreA = a.sourceDependents * (1.0 - a.similarity);
      const scoreB = b.sourceDependents * (1.0 - b.similarity);
      return scoreB - scoreA;
    });
  }
}

export interface Match {
  sourcePath: string;
  targetPath: string;
  sourceQualified: string;
  targetQualified: string;
  similarity: number;
  sourceDependents: number;
  targetDependents: number;
  sourceLines: number;
  targetLines: number;
  todoCount: number;
  lintCount: number;
  isStub: boolean;
  matchedByHeader: boolean;
  sourceFunctionCount: number;
  targetFunctionCount: number;
  matchedFunctionCount: number;
  functionCoverage: number;
  sourceTypeCount: number;
  targetTypeCount: number;
  matchedTypeCount: number;
  typeCoverage: number;
  missingTypes: string[];
  sourceDocLines: number;
  targetDocLines: number;
  sourceDocComments: number;
  targetDocComments: number;
  docSimilarity: number;
  docCoverage: number;
  docWeighted: number;
}
