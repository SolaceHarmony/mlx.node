// port-lint: source include/task_manager.hpp
import * as fs from "fs";
import * as path from "path";

/**
 * Task status for porting work items.
 */
export enum TaskStatus {
  PENDING = "pending",
  ASSIGNED = "assigned",
  COMPLETED = "completed",
  BLOCKED = "blocked",
}

/**
 * A single porting task.
 */
export interface PortTask {
  source_path: string;
  source_qualified: string;
  target_path: string;
  target_qualified: string;
  dependent_count: number;
  dependency_count: number;
  status: TaskStatus;
  assigned_to: string;
  assigned_at: string;
  completed_at: string;
  similarity: number;
  dependencies: string[];
  dependents: string[];
}

/**
 * RAII-like file lock for preventing race conditions.
 */
export class FileLock {
  private lockPath: string;
  private locked: boolean = false;

  constructor(filePath: string) {
    this.lockPath = filePath + ".lock";
    try {
      // advisory lock using a lock file
      fs.writeFileSync(this.lockPath, process.pid.toString(), { flag: "wx" });
      this.locked = true;
    } catch {
      this.locked = false;
    }
  }

  isLocked(): boolean {
    return this.locked;
  }

  unlock(): void {
    if (this.locked) {
      try {
        fs.unlinkSync(this.lockPath);
      } catch {
        // ignore
      }
      this.locked = false;
    }
  }
}

/**
 * Task file manager for coordinating swarm agents.
 */
export class TaskManager {
  task_file_path: string;
  agents_md_path: string = "";
  source_root: string = "";
  target_root: string = "";
  source_lang: string = "";
  target_lang: string = "";
  tasks: PortTask[] = [];

  constructor(taskFile: string) {
    this.task_file_path = taskFile;
  }

  load(): boolean {
    if (!fs.existsSync(this.task_file_path)) return false;
    try {
      const content = fs.readFileSync(this.task_file_path, "utf-8");
      const data = JSON.parse(content);

      this.source_root = data.source_root || "";
      this.target_root = data.target_root || "";
      this.source_lang = data.source_lang || "";
      this.target_lang = data.target_lang || "";
      this.agents_md_path = data.agents_md || "";
      this.tasks = data.tasks || [];

      return true;
    } catch (e) {
      console.error(`Error loading task file: ${(e as Error).message}`);
      return false;
    }
  }

  save(): boolean {
    try {
      const data = {
        source_root: this.source_root,
        target_root: this.target_root,
        source_lang: this.source_lang,
        target_lang: this.target_lang,
        agents_md: this.agents_md_path,
        tasks: this.tasks,
      };
      fs.writeFileSync(this.task_file_path, JSON.stringify(data, null, 2));
      return true;
    } catch (e) {
      console.error(`Error saving task file: ${(e as Error).message}`);
      return false;
    }
  }

  assignNext(agentId: string): PortTask | null {
    const lock = new FileLock(this.task_file_path);
    if (!lock.isLocked()) {
      console.warn("Warning: Could not acquire lock on task file");
      return null;
    }

    try {
      if (!this.load()) return null;

      const pending = this.tasks.filter((t) => t.status === TaskStatus.PENDING);
      if (pending.length === 0) return null;

      pending.sort((a, b) => b.dependent_count - a.dependent_count);

      const task = pending[0]!;
      task.status = TaskStatus.ASSIGNED;
      task.assigned_to = agentId;
      task.assigned_at = new Date().toISOString();

      if (!this.save()) {
        task.status = TaskStatus.PENDING;
        task.assigned_to = "";
        task.assigned_at = "";
        return null;
      }

      return task;
    } finally {
      lock.unlock();
    }
  }

  completeTask(sourceQualified: string): boolean {
    const lock = new FileLock(this.task_file_path);
    if (!lock.isLocked()) return false;

    try {
      if (!this.load()) return false;

      const task = this.tasks.find((t) => t.source_qualified === sourceQualified);
      if (task) {
        task.status = TaskStatus.COMPLETED;
        task.completed_at = new Date().toISOString();
        return this.save();
      }
      return false;
    } finally {
      lock.unlock();
    }
  }

  releaseTask(sourceQualified: string): boolean {
    const lock = new FileLock(this.task_file_path);
    if (!lock.isLocked()) return false;

    try {
      if (!this.load()) return false;

      const task = this.tasks.find(
        (t) => t.source_qualified === sourceQualified && t.status === TaskStatus.ASSIGNED
      );
      if (task) {
        task.status = TaskStatus.PENDING;
        task.assigned_to = "";
        task.assigned_at = "";
        return this.save();
      }
      return false;
    } finally {
      lock.unlock();
    }
  }

  getStats(): { pending: number; assigned: number; completed: number; blocked: number } {
    const stats = { pending: 0, assigned: 0, completed: 0, blocked: 0 };
    for (const t of this.tasks) {
      if (t.status === TaskStatus.PENDING) stats.pending++;
      else if (t.status === TaskStatus.ASSIGNED) stats.assigned++;
      else if (t.status === TaskStatus.COMPLETED) stats.completed++;
      else if (t.status === TaskStatus.BLOCKED) stats.blocked++;
    }
    return stats;
  }

  readAgentsMd(): string {
    if (!this.agents_md_path || !fs.existsSync(this.agents_md_path)) return "";
    return fs.readFileSync(this.agents_md_path, "utf-8");
  }

  printAssignment(task: PortTask, agentNumber: string | number): void {
    console.log("=== TASK ASSIGNMENT ===\n");
    console.log(`You are agent #${agentNumber}`);
    console.log(`Reminder: all ast_distance commands require: --agent ${agentNumber}\n`);

    const isTestTask = task.source_path.startsWith("tests/");

    let effectiveTargetRoot = this.target_root;
    if (this.target_lang === "kotlin" && isTestTask) {
      effectiveTargetRoot = effectiveTargetRoot.replace(/\/src\/commonMain\//g, "/src/commonTest/");
      effectiveTargetRoot = effectiveTargetRoot.replace(/src\/commonMain\//g, "src/commonTest/");
    }

    console.log("Source File:");
    console.log(`  Path:      ${this.source_root}/${task.source_path}`);
    console.log(`  Qualified: ${task.source_qualified}`);
    console.log(`  Dependents: ${task.dependent_count} files depend on this\n`);

    console.log("Target File:");
    console.log(`  Path:      ${effectiveTargetRoot}/${task.target_path}`);

    const commentPrefix = this.target_lang === "python" ? "##" : "//";
    let portLintPath = task.source_path;
    if (this.source_root.endsWith("/src") || this.source_root.endsWith("/src/")) {
      if (!portLintPath.startsWith("src/")) {
        portLintPath = "src/" + portLintPath;
      }
    }

    console.log(
      `  Add header: ${commentPrefix} port-lint: ${isTestTask ? "tests " : "source "}${portLintPath}\n`
    );

    console.log(`Priority: ${task.dependent_count} (higher = more critical)\n`);

    const agentsContent = this.readAgentsMd();
    if (agentsContent) {
      console.log("=== PORTING GUIDELINES (from AGENTS.md) ===\n");
      console.log(agentsContent);
      console.log("");
    }

    console.log("=== INSTRUCTIONS ===\n");
    console.log("1. Read the source file thoroughly");
    console.log("2. Create the target file at the target path");
    console.log("3. Add the port-lint header as the first line");
    console.log("4. Transliterate the source code to idiomatic target language");
    console.log("5. Match documentation comments from the source");
    console.log(
      `6. Run: ast_distance --agent ${agentNumber} <source> ${this.source_lang} <target> ${this.target_lang}`
    );
    console.log("   to verify similarity (aim for >0.85)");
    console.log(
      `7. When complete, run: ast_distance --agent ${agentNumber} --complete ${this.task_file_path} ${task.source_qualified}\n`
    );
  }
}
