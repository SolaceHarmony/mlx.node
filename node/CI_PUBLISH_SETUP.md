# Node Build and npm Publish Setup

This file documents the scaffolded CI and npm publish workflow for `mlx.node`.

## Build Method (local and CI)

Run from the repository root:

```bash
cd node
npm ci
npm run build:native
npm test
```

`build:native` runs `node-gyp configure build` and compiles `mlx.node`.

## GitHub Actions Scaffolding

- CI workflow: `.github/workflows/node-ci.yml`
  - Triggered on pull requests, pushes to `main`, and manual dispatch.
  - Runs on `macos-14` with Node 22.
  - Installs dependencies, builds native addon, and runs tests.

- Publish workflow: `.github/workflows/npm-publish.yml`
  - Manual dispatch only.
  - Supports `dist_tag` input (`latest`, `next`, etc.).
  - Supports `dry_run` input so you can validate before publishing.
  - Uses `NPM_TOKEN` from GitHub secrets.

## Required Secret

Add this secret in GitHub repository settings (or in the `npm-publish` environment):

- `NPM_TOKEN`: npm automation token with publish rights for the package.

Suggested path:

1. npm account: create an Automation token on npmjs.com.
2. GitHub repo: `Settings > Secrets and variables > Actions`.
3. Add `NPM_TOKEN` as a repository secret or environment secret.

## Important Package Metadata Check

Before first publish, verify package identity in `node/package.json`:

- `name`
- `version`
- `files`

If this package should publish as `mlx.node` (or a scoped package), update `name` before running publish workflow.
