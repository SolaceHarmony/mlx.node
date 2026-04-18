#!/usr/bin/env node

/**
 * Build script for AST Distance TypeScript package
 */

import { execSync } from "child_process";
import { existsSync, rmSync } from "fs";
import { join } from "path";

const ROOT = process.cwd();
const TYPESCRIPT_DIR = join(ROOT, "typescript");

const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
};

function log(message, color = "reset") {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function exec(command, cwd = ROOT) {
  try {
    log(`> ${command}`, "cyan");
    execSync(command, {
      cwd,
      stdio: "inherit",
      env: { ...process.env, FORCE_COLOR: "1" },
    });
  } catch (error) {
    log(`Error executing: ${command}`, "red");
    throw error;
  }
}

function clean() {
  const distDir = join(TYPESCRIPT_DIR, "dist");

  if (existsSync(distDir)) {
    log("Cleaning dist directory...", "yellow");
    rmSync(distDir, { recursive: true, force: true });
    log("Cleaned dist directory", "green");
  }
}

function install() {
  log("Installing dependencies...", "yellow");
  exec("npm ci", TYPESCRIPT_DIR);
}

function build() {
  log("Building TypeScript package...", "yellow");
  exec("npm run build", TYPESCRIPT_DIR);
}

function lint() {
  log("Running linter...", "yellow");
  exec("npm run lint", TYPESCRIPT_DIR);
}

function test() {
  log("Running tests...", "yellow");
  exec("npm test", TYPESCRIPT_DIR);
}

function main() {
  const command = process.argv[2] || "build";

  log(`AST Distance Build Script`, "bright");
  log(`Command: ${command}\n`, "cyan");

  switch (command) {
    case "clean":
      clean();
      break;

    case "install":
      clean();
      install();
      break;

    case "build":
      clean();
      build();
      log("\n✓ Build complete!", "green");
      break;

    case "lint":
      lint();
      break;

    case "test":
      build();
      test();
      break;

    case "ci":
    case "all":
      clean();
      install();
      lint();
      build();
      test();
      log("\n✓ CI complete!", "green");
      break;

    default:
      log(`Unknown command: ${command}`, "red");
      log("\nAvailable commands:", "bright");
      log("  clean    - Remove dist directory");
      log("  install  - Install dependencies");
      log("  build    - Build TypeScript package");
      log("  lint     - Run linter");
      log("  test     - Run tests");
      log("  ci/all   - Run full CI pipeline");
      process.exit(1);
  }
}

main();
