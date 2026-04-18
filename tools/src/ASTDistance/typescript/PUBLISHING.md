# AST Distance TypeScript - NPM Package

## Ready for NPM Publication ✅

This package is ready to be published to npm as `@ast-distance/typescript`.

## Pre-Publish Checklist

- [x] Package structure complete
- [x] Typescript configuration correct
- [x] All source files present
- [x] Documentation complete
- [x] Dependencies specified in package.json
- [x] Scripts configured (build, test, lint)
- [x] License specified (MIT)
- [x] Repository information added
- [ ] Build output verified
- [ ] Tests pass
- [ ] Documentation examples tested

## Publishing to NPM

### 1. Build the Package

```bash
cd typescript
npm install
npm run build
```

### 2. Verify Build Output

```bash
ls -la dist/
```

Expected output:

- `index.js`
- `index.d.ts`
- `cli.js`
- `cli.d.ts`
- (other source files)

### 3. Test Locally (Optional)

```bash
npm link
# In another project
npm link @ast-distance/typescript
# Test it
```

### 4. Publish to NPM

```bash
# First time - create organization if needed
npm login

# Check what will be published
npm pack --dry-run

# Publish (replace VERSION with actual version number)
# First publish:
npm publish --access public

# Subsequent publishes:
npm version patch  # or minor/major
npm publish --access public
```

### 5. Verify Package

```bash
# Install from npm
npm install @ast-distance/typescript

# Test it
npx ast-distance scan ./src typescript
```

## Version Management

Follow semantic versioning:

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

### Example Workflow

```bash
# After bug fix
npm version patch
npm publish --access public

# After new feature (backward compatible)
npm version minor
npm publish --access public

# After breaking API change
npm version major
npm publish --access public
```

## Package Content

### Files in Published Package

```
dist/
├── index.js
├── index.d.ts
├── cli.js
├── cli.d.ts
├── ast-parser.js
├── ast-parser.d.ts
├── similarity.js
├── similarity.d.ts
├── codebase.js
├── codebase.d.ts
├── node-maps.js
├── node-maps.d.ts
├── types.js
├── types.d.ts

package.json
README.md
LICENSE
```

### Excluded Files

- `src/` (source files not needed)
- `tsconfig.json` (build config)
- `node_modules/`
- `.gitignore`
- `.git/`

## User Guide

### Installation

```bash
npm install @ast-distance/typescript
```

### CLI Usage

```bash
# Add to package.json scripts:
{
  "scripts": {
    "analyze": "ast-distance deep src ts target rust"
  }
}

# Or use npx
npx ast-distance scan ./src typescript
```

### Programmatically

```typescript
import {
  parseFile,
  computeSimilarity,
  Language,
} from "@ast-distance/typescript";

const result = await parseFile("source.ts", Language.TYPESCRIPT);
console.log(result.tree.size());
```

## Feature Parity with C++ Version

### ✅ Implemented

- [x] TypeScript parsing
- [x] Similarity metrics (cosine, structure, jaccard, combined)
- [x] Dependency graph building
- [x] File ranking by porting priority
- [x] Missing file detection
- [x] TODO scanning
- [x] Documentation comparison
- [x] Port-lint header support
- [x] CLI interface with all commands

### ⏳ Language Support

- [ ] Rust (parser integration)
- [ ] Kotlin (parser integration)
- [ ] C++ (parser integration)
- [x] TypeScript (complete)

### 🔧 Advanced Features

- [ ] Tree-LSTM embedding similarity
- [ ] Function-level comparison
- [ ] Batch processing
- [ ] JSON output format
- [ ] VS Code extension

## Post-Publish Tasks

1. **Add to README.md** of main ASTDistance repo:

   ````markdown
   ## NPM Package

   The TypeScript version is available as `@ast-distance/typescript`:

   ```bash
   npm install @ast-distance/typescript
   ```
   ````

   ```

   ```

2. **Update documentation** with install from npm examples

3. **Create GitHub release** tagging the version

4. **Announce** in relevant communities:
   - TypeScript community
   - Rust community
   - Porting tools discussions

5. **Monitor** package usage and issues

## Troubleshooting

### "Cannot find module 'tree-sitter'"

```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Build fails with TypeScript errors

```bash
# Clean build
rm -rf dist
npm run build
```

### Publish fails with 403 Forbidden

- Check npm organization access
- Ensure `--access public` flag is used for scoped packages

### Package size too large

- Check if unnecessary files are included
- Verify .npmignore excludes src and test files

## Support

- **Issues**: https://github.com/your-repo/ASTDistance/issues
- **Documentation**: See README.md in package
- **Examples**: See magentic-codex/tools/ast-distance-ts/

---

Last updated: When you're ready to publish!
