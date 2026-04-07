"""Unified CLI entry point for Yorishiro.

Usage:
    python -m yorishiro run --project <dir> --source <id> --step film.audio
    python -m yorishiro run --project <dir> --source <id> --step film.audio.sound_events
    python -m yorishiro run --project <dir> --source <id> --step film
    python -m yorishiro run --project <dir> --source <id> --step novel.scenes --task ch003
    python -m yorishiro run --project <dir> --all
    python -m yorishiro run --project <dir> --source <id> --step film --force
    python -m yorishiro status --project <dir>
    python -m yorishiro list --project <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    from yorishiro.pipeline.orchestrator import CROSS_SOURCE_ID, Orchestrator, expand_step_prefix
    from yorishiro.project import Project

    project = Project.load(Path(args.project))
    orch = Orchestrator(project)

    if args.all:
        orch.run_all(force=args.force)
        return

    if args.group:
        if not args.source:
            group_steps = project.step_groups.get(args.group, [])
            cross_only = all(s.startswith("cross.") for s in group_steps)
            source_id = CROSS_SOURCE_ID if cross_only else _require_source(args)
        else:
            source_id = args.source
        orch.run_group(args.group, source_id, force=args.force)
        return

    if args.step:
        # Determine source_id: default to CROSS_SOURCE_ID only if all expanded
        # steps are cross steps; otherwise require --source.
        try:
            expanded = expand_step_prefix(args.step)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        all_cross = all(s.startswith("cross.") for s in expanded)
        source_id = CROSS_SOURCE_ID if all_cross else (args.source or _require_source(args))
        orch.run([args.step], source_id, force=args.force, task_key=args.task)
        return

    print("error: one of --step, --group, or --all is required", file=sys.stderr)
    sys.exit(1)


def _require_source(args: argparse.Namespace) -> str:
    if not args.source:
        print("error: --source is required for this step/group", file=sys.stderr)
        sys.exit(1)
    return args.source


def cmd_list(args: argparse.Namespace) -> None:
    from yorishiro.pipeline.orchestrator import ALL_STEPS, CROSS_SOURCE_ID, _build_step
    from yorishiro.project import Project
    from yorishiro.tasks.registry import ModelRegistry

    project = Project.load(Path(args.project))
    registry = ModelRegistry(project)

    print("Sources:")
    for s in project.sources:
        print(f"  {s.id}  ({s.type}, {s.authority})")

    print()
    print("Steps  (use prefix to run a group: --step film.audio, --step film, --step novel)")
    last_prefix: str = ""
    for step_id in ALL_STEPS:
        parts = step_id.split(".")
        scope = parts[0]
        if scope == "novel":
            source_ids = [s.id for s in project.sources if s.type == "novel"]
            applies = ", ".join(source_ids)
        elif scope == "film":
            source_ids = [s.id for s in project.sources if s.type == "film"]
            applies = ", ".join(source_ids)
        else:
            source_ids = [CROSS_SOURCE_ID]
            applies = "(cross-source)"

        # Print a blank separator line when the mid-level prefix changes
        mid_prefix = ".".join(parts[:2])
        if last_prefix and mid_prefix != last_prefix:
            print()
        last_prefix = mid_prefix

        # Probe task keys
        task_keys: list[str] = []
        if source_ids:
            try:
                step = _build_step(step_id, source_ids[0], project, registry)
                task_keys = [t.key for t in step.tasks() if t.key is not None]
            except Exception:
                pass

        if task_keys:
            print(f"  {step_id:<28} {applies}  ({len(task_keys)} tasks: --task {task_keys[0]} … {task_keys[-1]})")
        else:
            print(f"  {step_id:<28} {applies}")

    if project.step_groups:
        print()
        print("Named groups  (--group <name>):")
        for name, steps in project.step_groups.items():
            print(f"  {name:<20} {', '.join(steps)}")


def cmd_status(args: argparse.Namespace) -> None:
    from yorishiro.pipeline.orchestrator import Orchestrator
    from yorishiro.project import Project

    project = Project.load(Path(args.project))
    orch = Orchestrator(project)
    orch.status()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="yorishiro",
        description="Yorishiro — character soul document generator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = sub.add_parser("run", help="Run one or more pipeline steps")
    run_parser.add_argument("--project", required=True, help="Path to project directory")
    run_parser.add_argument("--source", help="Source ID (e.g. cpk-novel)")
    run_parser.add_argument("--step", help="Step ID (e.g. novel.scenes)")
    run_parser.add_argument("--task", help="Task key within a step (e.g. ch003)")
    run_parser.add_argument("--group", help="Named step group (e.g. novel-full)")
    run_parser.add_argument("--all", action="store_true", help="Run all steps for all sources")
    run_parser.add_argument("--force", action="store_true", help="Re-run even if up to date")

    # --- list ---
    list_parser = sub.add_parser("list", help="List sources, steps, and step groups")
    list_parser.add_argument("--project", required=True, help="Path to project directory")

    # --- status ---
    status_parser = sub.add_parser("status", help="Show completion status for all steps")
    status_parser.add_argument("--project", required=True, help="Path to project directory")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
