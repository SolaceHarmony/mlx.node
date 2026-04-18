# AST Distance - Semantic Awareness Improvements

## Overview

Three-phase enhancement to eliminate "semantic blindness" and improve porting quality verification.

## Problem Statement

The original `ast_distance` tool suffered from three critical blind spots:

### 1. The "Unnamed Node" Blind Spot
**Problem:** Operators (+, -, *, &&, ||) are unnamed nodes in Tree-Sitter and were ignored.
**Impact:** `a + b` and `a * b` produced identical AST histograms → appeared 100% similar
**Result:** Logic errors completely masked

### 2. Unweighted Cosine Similarity  
**Problem:** All node types weighted equally in similarity calculation
**Impact:** Missing CLASS_DECLARATION penalized same as missing LPAREN
**Result:** Common nodes (identifiers, blocks) drowned out structural signals

### 3. The "UNKNOWN" Bucket
**Problem:** Binary expressions broadly mapped to NodeType::UNKNOWN
**Impact:** All binary logic flattened, semantic differences invisible
**Result:** Cannot distinguish different operator semantics

## Solutions Implemented

### Phase 1: Operational Awareness (Operator Capture)

**Changes:**
- Parse unnamed operator tokens as distinct nodes
- Added 5 new NodeType enums:
  - `ARITHMETIC_OP` (+, -, *, /, %)
  - `COMPARISON_OP` (==, !=, <, >, <=, >=)
  - `LOGICAL_OP` (&&, ||, !)
  - `BITWISE_OP` (&, |, ^, ~, <<, >>)
  - `ASSIGNMENT_OP` (=, +=, -=, etc.)

**Impact:**
- `a + b` vs `a * b` now correctly distinguished
- Operator-level semantic differences detected
- Logic errors no longer masked

**Code Location:** `include/ast_parser.hpp:403-467`, `include/node_types.hpp:75-79`

### Phase 2: Signal Boosting (Weighted Similarity)

**Changes:**
- Implemented `get_node_weight()` with heuristic importance weights:
  ```cpp
  CLASS, FUNCTION, IF, WHILE, etc.  → 5.0x weight
  CALL, RETURN, THROW, LAMBDA       → 2.0x weight  
  Operators (new from Phase 1)      → 1.5x weight
  VARIABLE, VAR_DECL                → 0.5x weight
  Default                           → 1.0x weight
  ```
- Applied weights in `histogram_cosine_similarity()`

**Impact:**
- Structural mismatches properly tank similarity scores
- Missing CLASS now has 10x impact vs missing VARIABLE
- Semantic signal no longer drowned by boilerplate

**Code Location:** `include/similarity.hpp:23-79`

### Phase 3: Token Histograms (Identifier Analysis)

**Changes:**
- Added `IdentifierStats` struct for tracking identifier frequencies
- Implemented `extract_identifiers()` method
- `identifier_cosine_similarity()` for naming divergence detection
- Filters out boilerplate (`it`, `this`, single-char names)

**Impact:**
- Detects naming inconsistencies: `collect()` vs `gather()`
- Secondary similarity vector for token-level analysis
- Flags semantic equivalence with different naming conventions

**Code Location:** `include/ast_parser.hpp:30-78`, `266-335`, `471-502`

## Quality Control Enhancement

### Task Release Enforcement

**Feature:** Similarity check before allowing task release
**Threshold:** Requires >= 0.70 similarity to release assigned task
**Benefits:**
- Prevents abandoning incomplete ports
- Hard fails on syntax errors (unparseable files)
- No silent failures - all errors block release

**Code Location:** `src/main.cpp:1140-1183`

## Testing Results

### kotlin.coroutines-cpp (Kotlin→C++)

**Before (Unweighted) vs After (Weighted):**

| File | Old | New | Change | Interpretation |
|------|-----|-----|--------|----------------|
| intrinsics.Intrinsics | 0.61 | **0.53** | -13% | Structural differences properly penalized |
| ContinuationInterceptor | 0.66 | **0.59** | -11% | Now flagged as incomplete |
| Continuation | 0.75 | **0.65** | -13% | Semantic drift detected |
| CoroutineContextImpl | 0.71 | **0.68** | -4% | Slightly stricter |

**Improved Detection:**
- Old: 2 files < 0.60 (incomplete)
- New: **4 files < 0.60** (incomplete)
- ✅ Correctly identifies 2 additional problem files

## Deployment Status

- ✅ **ASTDistance** (master branch - all 3 phases)
- ✅ **kotlin.coroutines-cpp** (committed with all phases)
- ✅ **codex-kotlin** (committed with all phases)
- ✅ **starlark-kotlin** (binary updated)
- ✅ **TypeScript/Node.js** (v0.2.0 with Phases 1-2)

## Usage

No API changes required. All improvements are automatic in:
- Single file comparison: `ast_distance file1.kt kotlin file2.cpp cpp`
- Deep mode analysis: `ast_distance --deep src/kotlin kotlin src/cpp cpp`
- Task management: `ast_distance --release tasks.json <task>`

## Future Enhancements

Potential Phase 4+ improvements:
1. **UNKNOWN Bucket Cleanup** - Map specific expression types
2. **Deep Mode Reports** - Better critical file identification
3. **Doc Gap Analysis** - KDoc→Doxygen suggestions
4. **Performance** - Caching, parallel processing
5. **Identifier Integration** - Add to --deep mode reports

## Credits

- Phase 1-3 design and implementation: 2026-01-30
- Tested on kotlin.coroutines-cpp port
- Deployed across 4 projects
