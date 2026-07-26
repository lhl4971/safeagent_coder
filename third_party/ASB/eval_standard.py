"""
ASB Standard Defense Evaluation Script (Baseline / LlamaGuard / LLMGuard)

Evaluates defense mechanisms against prompt injection attacks (DPI, IPI)
and memory poisoning (MP) on the ASB benchmark suite.

Usage:
    python third_party/ASB/eval_standard.py --defense baseline \\
        --dpi --ipi --attack-start 0 --attack-end 39 \\
        --output outputs/asb_results_baseline.csv

    python third_party/ASB/eval_standard.py --defense llama_guard \\
        --ipi --attack-start 0 --attack-end 39 \\
        --output outputs/asb_results_llama_guard.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from third_party.ASB.backbone import (
    ASBAgentConfig,
    attack_happened,
    judge_task_success,
    load_agent_config,
    load_agent_tasks,
    load_attacker_tools_from_jsonl,
    load_tool_specs_from_jsonl,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "agent_name", "task", "attacker_idx",
    "attack_ok", "task_ok", "both_ok",
    "status", "error_type", "error_message",
]

_DEFENSE_MIDDLEWARES: Dict[str, str] = {
    "baseline": None,
    "llama_guard": "third_party.llama.backbone",
    "llm_guard": "third_party.llm_guard.backbone",
}


# ---------------------------------------------------------------------------
# Agent cache
# ---------------------------------------------------------------------------
class _AgentCache:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


_AGENT_CACHE: Dict[str, _AgentCache] = {}


def _get_agent_cache(agent_name: str) -> _AgentCache:
    if agent_name in _AGENT_CACHE:
        return _AGENT_CACHE[agent_name]
    agent_cfg: ASBAgentConfig = load_agent_config(agent_name)
    system_prompt = agent_cfg.build_system_prompt()
    _AGENT_CACHE[agent_name] = _AgentCache(system_prompt)
    return _AGENT_CACHE[agent_name]


def _build_normal_tools(agent_name: str, injection: Optional[str] = None) -> list:
    normal_specs = load_tool_specs_from_jsonl(agent_name)
    return [t.to_langchain_tool(injection) for t in normal_specs]


# ---------------------------------------------------------------------------
# Middleware loading
# ---------------------------------------------------------------------------
def load_middlewares(defense: str) -> list:
    """Load model-level middleware for the given defense strategy."""
    if defense == "baseline":
        return []
    elif defense == "llama_guard":
        from third_party.llama.backbone import (  # type: ignore[import-untyped]
            llama_guard_after_agent,
            llama_guard_before_model,
        )
        return [llama_guard_before_model, llama_guard_after_agent]
    elif defense == "llm_guard":
        from third_party.llm_guard.backbone import (  # type: ignore[import-untyped]
            llm_guard_after_agent,
            llm_guard_before_model,
        )
        return [llm_guard_before_model, llm_guard_after_agent]
    else:
        raise ValueError(f"Unknown defense: {defense}")


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------
def run_one_case(
    agent_name: str,
    task: str,
    attacker_idx: Optional[int],
    base_model: ChatOpenAI,
    judge_model: ChatOpenAI,
    middlewares: list,
    *,
    direct_injection: bool,
    indirect_injection: bool,
    memory_poisoning: bool,
) -> Tuple[bool, bool]:
    """Run one ASB test case. Returns (attack_ok, task_ok)."""

    cache = _get_agent_cache(agent_name)

    attacker: Optional[Any] = None
    attacker_tool_name: Optional[str] = None
    injection_payload: Optional[str] = None

    # --- load attacker spec ---
    if attacker_idx is not None and attacker_idx >= 0:
        attacker_specs = load_attacker_tools_from_jsonl(
            agent_name, attacker_indices=[attacker_idx]
        )
        if not attacker_specs:
            raise RuntimeError(
                f"No attacker tool for agent={agent_name}, idx={attacker_idx}"
            )
        attacker = attacker_specs[0]
        attacker_tool_name = attacker.tool_name

        if indirect_injection:
            injection_payload = attacker.build_injection_prompt()

    # --- build normal tools ---
    tools = _build_normal_tools(
        agent_name,
        injection=injection_payload if indirect_injection else None,
    )

    # --- optionally add attacker tool ---
    if attacker is not None:
        tools.append(attacker.to_langchain_tool())

    # --- create agent (middleware must be a list, never None) ---
    agent = create_agent(
        model=base_model,
        tools=tools,
        system_prompt=cache.system_prompt,
        middleware=middlewares or [],
    )

    # --- DPI: rewrite user query ---
    user_query = task
    if attacker is not None and direct_injection:
        user_query = f"{task} {attacker.build_injection_prompt()}"

    # --- memory poisoning ---
    if attacker is not None and memory_poisoning:
        messages: List[BaseMessage] = [
            AIMessage(attacker.build_poisoned_message(task))
        ]
    else:
        messages = []

    messages.append(HumanMessage(content=user_query))
    result = agent.invoke({"messages": messages})
    msgs: List[BaseMessage] = (
        result.get("messages", []) if isinstance(result, dict) else []
    )

    attack_ok = (
        attack_happened(msgs, attacker_tool_name) if attacker_tool_name else False
    )
    task_ok = bool(judge_task_success(msgs, task, judge_model))
    return attack_ok, task_ok


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def load_completed_jobs(csv_path: str) -> set:
    """Read existing CSV and return a set of (agent_name, task, attacker_idx)."""
    if not Path(csv_path).exists():
        return set()
    existing = pd.read_csv(csv_path)
    return set(zip(existing["agent_name"], existing["task"], existing["attacker_idx"]))


def write_header(csv_path: str) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------
def run_evaluation(
    *,
    defense: str,
    output_csv: str,
    attack_start: int = 0,
    attack_end: int = -1,
    direct_injection: bool = False,
    indirect_injection: bool = False,
    memory_poisoning: bool = False,
    max_workers: int = 64,
    resume: bool = True,
    benign: bool = False,
) -> None:
    """Run the full ASB evaluation sweep with CSV output and resume support."""

    # --- models ---
    base_model = ChatOpenAI(
        model="deepseek/deepseek-v3.2",
        base_url=os.environ["SAFEGUARD_ENDPOINT_URL"],
        api_key=os.environ["SAFEGUARD_ENDPOINT_API_KEY"],
        temperature=0,
    )
    judge_model = ChatOpenAI(
        model="deepseek/deepseek-v3.2",
        base_url=os.environ["SAFEGUARD_ENDPOINT_URL"],
        api_key=os.environ["SAFEGUARD_ENDPOINT_API_KEY"],
        temperature=0,
    )

    # --- middlewares ---
    middlewares = load_middlewares(defense)
    print(f"[eval] Defense mode: {defense}")
    print(f"[eval] Benign={benign}")
    if not benign:
        print(f"[eval] DPI={direct_injection}, IPI={indirect_injection}, MP={memory_poisoning}")
    print(f"[eval] Middlewares: {len(middlewares)}")

    # --- build job list ---
    cases = load_agent_tasks()

    if benign:
        # Benign: one job per (agent, task), no attacker
        all_jobs: List[Tuple[str, str, int]] = [
            (item["agent_name"], item["task"], -1) for item in cases
        ]
    else:
        idx_list = list(range(attack_start, attack_end + 1))
        all_jobs = []
        for item in cases:
            for idx in idx_list:
                all_jobs.append((item["agent_name"], item["task"], idx))

    print(f"[eval] Total tasks: {len(all_jobs)} "
          f"({len(cases)} agents × {'benign' if benign else str(len(idx_list)) + ' attackers'})")

    # --- resume logic ---
    if resume and Path(output_csv).exists():
        done = load_completed_jobs(output_csv)
        print(f"[eval] Existing CSV: {len(done)} completed tasks found")
        jobs = [j for j in all_jobs if j not in done]
        print(f"[eval] Remaining: {len(jobs)} (skipped {len(all_jobs) - len(jobs)})")
    else:
        jobs = all_jobs
        write_header(output_csv)
        print(f"[eval] Fresh run: {len(jobs)} tasks, CSV header written")

    if not jobs:
        print("[eval] All tasks already completed. Nothing to do.")
        return

    # --- execution ---
    csv_lock = threading.Lock()

    def append_record(record: Dict) -> None:
        with csv_lock:
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(record)

    total = len(jobs)
    attack_succ = task_succ = both_succ = err_count = 0
    pbar = tqdm(total=total, desc=f"ASB ({defense})")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                run_one_case,
                agent_name,
                task,
                idx,
                base_model,
                judge_model,
                middlewares,
                direct_injection=direct_injection,
                indirect_injection=indirect_injection,
                memory_poisoning=memory_poisoning,
            ): (agent_name, task, idx)
            for (agent_name, task, idx) in jobs
        }

        for fut in as_completed(futures):
            agent_name, task, idx = futures[fut]
            try:
                a_ok, t_ok = fut.result()
                record = {
                    "agent_name": agent_name,
                    "task": task,
                    "attacker_idx": idx,
                    "attack_ok": int(a_ok),
                    "task_ok": int(t_ok),
                    "both_ok": int(a_ok and t_ok),
                    "status": "ok",
                    "error_type": "",
                    "error_message": "",
                }
                attack_succ += int(a_ok)
                task_succ += int(t_ok)
                both_succ += int(a_ok and t_ok)
            except Exception as e:
                record = {
                    "agent_name": agent_name,
                    "task": task,
                    "attacker_idx": idx,
                    "attack_ok": 0,
                    "task_ok": 0,
                    "both_ok": 0,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                err_count += 1
                tqdm.write(f"[WARN] {agent_name} idx={idx}: {type(e).__name__}: {e}")

            append_record(record)
            pbar.update(1)

    pbar.close()

    # --- summary ---
    n_ok = total - err_count
    asr = attack_succ / total if total else 0.0
    tsr = task_succ / total if total else 0.0
    bsr = both_succ / total if total else 0.0

    print()
    print("=" * 60)
    print(f"  Defense:     {defense}")
    print(f"  Tests:       {total}  (OK: {n_ok}, ERR: {err_count})")
    print("-" * 60)
    print(f"  ASR:         {asr:.4f}  ({attack_succ}/{total})")
    print(f"  TSR:         {tsr:.4f}  ({task_succ}/{total})")
    print(f"  BSR:         {bsr:.4f}  ({both_succ}/{total})")
    print("=" * 60)
    print(f"[eval] Results written to: {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASB Standard Defense Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--defense", type=str, default="baseline",
        choices=["baseline", "llama_guard", "llm_guard"],
        help="Defense strategy to evaluate (default: baseline)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output CSV path (default: outputs/asb_results_{defense}.csv)",
    )

    attack_group = parser.add_argument_group("Attack settings")
    attack_group.add_argument(
        "--attack-start", type=int, default=0,
        help="First attacker index (inclusive, default: 0)",
    )
    attack_group.add_argument(
        "--attack-end", type=int, default=39,
        help="Last attacker index (inclusive, default: 39)",
    )

    injection_group = parser.add_argument_group("Injection types")
    injection_group.add_argument(
        "--dpi", action="store_true", default=False,
        help="Enable direct prompt injection (DPI)",
    )
    injection_group.add_argument(
        "--ipi", action="store_true", default=True,
        help="Enable indirect prompt injection (IPI, default: True)",
    )
    injection_group.add_argument(
        "--mp", action="store_true", default=False,
        help="Enable memory poisoning (MP)",
    )

    parser.add_argument(
        "--max-workers", type=int, default=64,
        help="Thread pool size (default: 64)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", default=False,
        help="Disable resume: overwrite existing CSV and start fresh",
    )

    # Benign mode
    parser.add_argument(
        "--benign", action="store_true", default=False,
        help="Run benign tasks only (no attacker, measures PNA via task_ok)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default output path
    if args.output is None:
        suffix = "benign" if args.benign else args.defense
        args.output = f"outputs/asb_results_{suffix}.csv"

    run_evaluation(
        defense=args.defense,
        output_csv=args.output,
        attack_start=args.attack_start,
        attack_end=args.attack_end,
        direct_injection=args.dpi,
        indirect_injection=args.ipi,
        memory_poisoning=args.mp,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        benign=args.benign,
    )


if __name__ == "__main__":
    main()
