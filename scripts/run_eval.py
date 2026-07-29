#!/usr/bin/env python3
"""
ASB Evaluation Entry Point
==========================
Reads eval_cases.yaml (same directory as this script) and dispatches each
test case to the appropriate evaluation script
(third_party/ASB/eval_standard.py or eval_safeagent.py).

Usage:
    python script/run_eval.py                          # run all cases
    python script/run_eval.py --case baseline           # run one case
    python script/run_eval.py --cases baseline,safeagent  # specific cases
    python script/run_eval.py --list                    # list defined cases
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


# Project root = directory containing script/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "eval_cases.yaml"


def load_cases(path: Path = CONFIG_PATH) -> List[Dict[str, Any]]:
    """Load test case definitions from YAML."""
    if not path.exists():
        print(f"[ERROR] Config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = raw.get("cases", [])
    if not cases:
        print(f"[ERROR] No cases defined in {path}", file=sys.stderr)
        sys.exit(1)
    return cases


def is_safeagent(defense: str) -> bool:
    """Detect whether a defense uses the SafeAgent (MCP) eval script."""
    return "safeagent" in defense.lower()


def build_command(case: Dict[str, Any]) -> List[str]:
    """
    Build the CLI argument list for a single test case.
    Automatically detects whether to use eval_standard.py or eval_safeagent.py.
    """
    defense: str = case["defense"]
    benign: bool = case.get("benign", False)
    output: str = case["output"]
    mcp_url: str = case.get("mcp_url", "")

    # Choose script (paths relative to project root)
    if is_safeagent(defense):
        script = PROJECT_ROOT / "third_party/ASB/eval_safeagent.py"
    else:
        script = PROJECT_ROOT / "third_party/ASB/eval_standard.py"

    cmd = [
        sys.executable,
        str(script),
        "--output", output,
    ]

    # Only eval_standard.py has a --defense argument
    if not is_safeagent(defense):
        cmd.extend(["--defense", defense])

    if benign:
        cmd.append("--benign")
        return cmd

    # Attack-specific parameters
    injections: list = case.get("injections", [])
    for inj in injections:
        inj_lower = inj.lower()
        if inj_lower == "dpi":
            cmd.append("--dpi")
        elif inj_lower == "ipi":
            cmd.append("--ipi")
        elif inj_lower == "mp":
            cmd.append("--mp")

    cmd.extend([
        "--attack-start", str(case.get("attack_start", 0)),
        "--attack-end", str(case.get("attack_end", -1)),
    ])

    if is_safeagent(defense) and mcp_url:
        cmd.extend(["--mcp-url", mcp_url])

    return cmd


def run_case(case: Dict[str, Any], dry_run: bool = False) -> int:
    """Execute one test case. Returns the subprocess return code."""
    cmd = build_command(case)
    defense = case["defense"]
    benign = case.get("benign", False)
    label = f"{defense} (benign)" if benign else defense
    output = case["output"]

    print()
    print("=" * 70)
    print(f"  Case: {label}")
    print(f"  Output: {output}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70)
    print()

    if dry_run:
        return 0

    import os

    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return result.returncode


def list_cases(cases: List[Dict[str, Any]]) -> None:
    """Print a summary of all defined test cases."""
    print(f"\n{'Defense':<20} {'Type':<10} {'Injections':<20} {'Attack Range':<15} {'Output':<40}")
    print("-" * 110)
    for c in cases:
        defense = c["defense"]
        benign = c.get("benign", False)
        if benign:
            ctype = "benign"
            injections = "-"
            attack_range = "-"
        else:
            ctype = "attack"
            injections = ", ".join(c.get("injections", []))
            attack_range = f"{c.get('attack_start', 0)}–{c.get('attack_end', -1)}"
        print(f"{defense:<20} {ctype:<10} {injections:<20} {attack_range:<15} {c['output']:<40}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASB Evaluation Entry Point — run test cases from eval_cases.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case", "-c", type=str, default=None,
        help="Run a single case by defense name (e.g. 'baseline', 'safeagent')",
    )
    parser.add_argument(
        "--cases", type=str, default=None,
        help="Comma-separated list of defense names to run",
    )
    parser.add_argument(
        "--list", "-l", action="store_true", default=False,
        help="List all defined test cases and exit",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", default=False,
        help="Print commands without executing",
    )
    parser.add_argument(
        "--config", type=str, default=str(CONFIG_PATH),
        help=f"Path to YAML config (default: {CONFIG_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.config))

    if args.list:
        list_cases(cases)
        return

    # Filter cases
    if args.case:
        filtered = [c for c in cases if c["defense"] == args.case]
        if not filtered:
            print(f"[ERROR] No case found with defense='{args.case}'", file=sys.stderr)
            sys.exit(1)
    elif args.cases:
        names = [x.strip() for x in args.cases.split(",")]
        filtered = [c for c in cases if c["defense"] in names]
        if not filtered:
            print(f"[ERROR] No cases matched: {args.cases}", file=sys.stderr)
            sys.exit(1)
    else:
        filtered = cases

    # Execute
    exit_codes = []
    for case in filtered:
        code = run_case(case, dry_run=args.dry_run)
        exit_codes.append(code)

    failed = [(case["defense"], code) for case, code in zip(filtered, exit_codes) if code != 0]
    if failed:
        print(f"\n[SUMMARY] {len(failed)} / {len(filtered)} cases FAILED:")
        for name, code in failed:
            print(f"  - {name} (exit code {code})")
        sys.exit(1)
    else:
        print(f"\n[SUMMARY] All {len(filtered)} cases completed successfully.")


if __name__ == "__main__":
    main()
