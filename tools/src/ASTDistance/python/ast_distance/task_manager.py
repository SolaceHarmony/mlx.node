"""Task manager for coordinating swarm agents (Python implementation).

The Python CLI (`ast_distance.cli`) expects this module to provide:
  - `PortTask` as a keyword-constructible type (dataclass)
  - `TaskManager.get_stats()` returning a dict
  - `TaskManager.format_assignment()` returning printable instructions

Older transliterations matched the C++ header shape but drifted from the Python CLI.
This module is the source of truth for the Python package.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TaskStatus(Enum):
    """Task status for porting work items."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class FileLock:
    """Advisory file lock (flock) scoped to this object lifetime."""

    def __init__(self, path: str):
        self.fd = -1
        self.locked = False
        self.lock_path = path + ".lock"

        try:
            self.fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            self.locked = True
        except (OSError, IOError):
            self.locked = False

    def __del__(self):
        if self.fd < 0:
            return
        try:
            if self.locked:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        except (OSError, IOError):
            pass
        try:
            os.close(self.fd)
        except (OSError, IOError):
            pass

    def is_locked(self) -> bool:
        return self.locked


@dataclass
class PortTask:
    """A single porting task entry in `tasks.json`."""

    source_path: str
    source_qualified: str
    target_path: str

    target_qualified: str = ""
    dependent_count: int = 0
    dependency_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str = ""
    assigned_at: str = ""
    completed_at: str = ""
    similarity: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "PortTask":
        status_raw = str(obj.get("status", TaskStatus.PENDING.value))
        try:
            status = TaskStatus(status_raw)
        except ValueError:
            status = TaskStatus.PENDING

        return PortTask(
            source_path=str(obj.get("source_path", "")),
            source_qualified=str(obj.get("source_qualified", "")),
            target_path=str(obj.get("target_path", "")),
            target_qualified=str(obj.get("target_qualified", "")),
            dependent_count=int(obj.get("dependent_count", 0) or 0),
            dependency_count=int(obj.get("dependency_count", 0) or 0),
            status=status,
            assigned_to=str(obj.get("assigned_to", "")),
            assigned_at=str(obj.get("assigned_at", "")),
            completed_at=str(obj.get("completed_at", "")),
            similarity=float(obj.get("similarity", 0.0) or 0.0),
            dependencies=list(obj.get("dependencies", []) or []),
            dependents=list(obj.get("dependents", []) or []),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_path": self.source_path,
            "source_qualified": self.source_qualified,
            "target_path": self.target_path,
            "dependent_count": self.dependent_count,
            "dependency_count": self.dependency_count,
            "status": self.status.value,
        }

        if self.target_qualified:
            out["target_qualified"] = self.target_qualified
        if self.assigned_to:
            out["assigned_to"] = self.assigned_to
        if self.assigned_at:
            out["assigned_at"] = self.assigned_at
        if self.completed_at:
            out["completed_at"] = self.completed_at
        if self.similarity:
            out["similarity"] = self.similarity
        if self.dependencies:
            out["dependencies"] = list(self.dependencies)
        if self.dependents:
            out["dependents"] = list(self.dependents)

        return out


class TaskManager:
    """Task file manager for coordinating swarm agents."""

    def __init__(self, task_file: str):
        self.task_file_path: str = task_file
        self.agents_md_path: str = ""
        self.source_root: str = ""
        self.target_root: str = ""
        self.source_lang: str = ""
        self.target_lang: str = ""
        self.tasks: list[PortTask] = []

    def load(self) -> bool:
        try:
            data = json.loads(Path(self.task_file_path).read_text(encoding="utf-8"))
        except Exception:
            return False

        self.source_root = str(data.get("source_root", ""))
        self.target_root = str(data.get("target_root", ""))
        self.source_lang = str(data.get("source_lang", ""))
        self.target_lang = str(data.get("target_lang", ""))
        self.agents_md_path = str(data.get("agents_md", ""))

        self.tasks = []
        tasks_raw = data.get("tasks", []) or []
        for item in tasks_raw:
            if not isinstance(item, dict):
                continue
            task = PortTask.from_json(item)
            if task.source_path:
                self.tasks.append(task)

        return True

    def save(self) -> bool:
        try:
            payload = {
                "source_root": self.source_root,
                "target_root": self.target_root,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "agents_md": self.agents_md_path,
                "tasks": [t.to_json() for t in self.tasks],
            }
            Path(self.task_file_path).write_text(
                json.dumps(payload, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
            return True
        except Exception:
            return False

    def assign_next(self, agent_id: str) -> Optional[PortTask]:
        lock = FileLock(self.task_file_path)
        if not lock.is_locked():
            import sys

            print("Warning: Could not acquire lock on task file", file=sys.stderr)
            return None

        if not self.load():
            import sys

            print("Warning: Could not reload task file", file=sys.stderr)
            return None

        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        if not pending:
            return None

        pending.sort(key=lambda t: t.dependent_count, reverse=True)
        task = pending[0]

        task.status = TaskStatus.ASSIGNED
        task.assigned_to = agent_id
        task.assigned_at = self._current_timestamp()

        if not self.save():
            import sys

            print("Warning: Could not save task file after assignment", file=sys.stderr)
            task.status = TaskStatus.PENDING
            task.assigned_to = ""
            task.assigned_at = ""
            return None

        return task

    def complete_task(self, source_qualified: str) -> bool:
        lock = FileLock(self.task_file_path)
        if not lock.is_locked():
            import sys

            print("Warning: Could not acquire lock for complete_task", file=sys.stderr)
            return False

        if not self.load():
            return False

        for t in self.tasks:
            if t.source_qualified == source_qualified:
                t.status = TaskStatus.COMPLETED
                t.completed_at = self._current_timestamp()
                return self.save()
        return False

    def release_task(self, source_qualified: str) -> bool:
        lock = FileLock(self.task_file_path)
        if not lock.is_locked():
            import sys

            print("Warning: Could not acquire lock for release_task", file=sys.stderr)
            return False

        if not self.load():
            return False

        for t in self.tasks:
            if t.source_qualified == source_qualified and t.status == TaskStatus.ASSIGNED:
                t.status = TaskStatus.PENDING
                t.assigned_to = ""
                t.assigned_at = ""
                return self.save()
        return False

    def get_stats(self) -> dict[str, int]:
        pending = assigned = completed = blocked = 0
        for t in self.tasks:
            if t.status == TaskStatus.PENDING:
                pending += 1
            elif t.status == TaskStatus.ASSIGNED:
                assigned += 1
            elif t.status == TaskStatus.COMPLETED:
                completed += 1
            elif t.status == TaskStatus.BLOCKED:
                blocked += 1
        return {
            "pending": pending,
            "assigned": assigned,
            "completed": completed,
            "blocked": blocked,
        }

    def read_agents_md(self) -> str:
        if not self.agents_md_path:
            return ""
        try:
            return Path(self.agents_md_path).read_text(encoding="utf-8")
        except Exception:
            return ""

    def format_assignment(self, task: PortTask, agent_number: int) -> str:
        prefix = self._port_lint_comment_prefix(self.target_lang)
        kind = "tests" if self._is_test_task(task) else "source"
        header = f"{prefix} port-lint: {kind} {task.source_path}"

        lines: list[str] = []
        lines.append("=== TASK ASSIGNMENT ===\n")
        lines.append(f"You are agent #{agent_number}")
        lines.append(f"Reminder: all ast_distance commands require: --agent {agent_number}\n")

        lines.append("Source File:")
        lines.append(f"  Path:      {self.source_root}/{task.source_path}")
        lines.append(f"  Qualified: {task.source_qualified}")
        lines.append(f"  Dependents: {task.dependent_count} files depend on this\n")

        lines.append("Target File:")
        lines.append(f"  Path:      {self.target_root}/{task.target_path}")
        lines.append(f"  Add header: {header}\n")

        lines.append(f"Priority: {task.dependent_count} (higher = more critical)\n")

        agents_content = self.read_agents_md()
        if agents_content:
            lines.append("=== PORTING GUIDELINES (from AGENTS.md) ===\n")
            lines.append(agents_content.rstrip() + "\n")

        lines.append("=== INSTRUCTIONS ===\n")
        lines.append("1. Read the source file thoroughly")
        lines.append("2. Create the target file at the target path")
        lines.append("3. Add the port-lint header as the first line")
        lines.append(f"4. Transliterate the source code to idiomatic {self.target_lang}")
        lines.append("5. Match documentation comments from the source")
        lines.append(
            f"6. Run: ast_distance --agent {agent_number} <source> {self.source_lang} <target> {self.target_lang}"
        )
        lines.append("   to verify similarity (aim for >0.85)")
        lines.append(
            f"7. When complete, run: ast_distance --agent {agent_number} --complete {self.task_file_path} {task.source_qualified}\n"
        )

        return "\n".join(lines)

    @staticmethod
    def _port_lint_comment_prefix(lang: str) -> str:
        if lang == "python":
            return "##"
        return "//"

    @staticmethod
    def _is_test_task(task: PortTask) -> bool:
        p = Path(task.target_path)

        for part in p.parts:
            if part in {"test", "tests"}:
                return True
            if part.lower().endswith("test") or part.lower().endswith("tests"):
                return True

        if any(
            s in task.target_path
            for s in (
                "/commonTest/",
                "/jvmTest/",
                "/jsTest/",
                "/nativeTest/",
                "/androidTest/",
                "/iosTest/",
                "/macosTest/",
                "/linuxTest/",
                "/mingwTest/",
                "/wasmTest/",
                "/wasmJsTest/",
            )
        ):
            return True

        if "/test/" in task.source_path or "/tests/" in task.source_path:
            return True

        return False

    @staticmethod
    def _current_timestamp() -> str:
        return datetime.now().isoformat()
