"""
ASB SafeAgent Defense Evaluation Script

Evaluates the SafeAgent defense mechanism (MCP-based) against prompt injection
attacks (DPI, IPI) and memory poisoning (MP) on the ASB benchmark suite.

Requires a running SafeAgent MCP core server (default: http://127.0.0.1:8000/mcp).

Usage:
    python third_party/ASB/eval_safeagent.py \\
        --ipi --attack-start 0 --attack-end 39 \\
        --output outputs/asb_results_safe_IPI.csv --label IPI

    python third_party/ASB/eval_safeagent.py \\
        --mp --attack-start 0 --attack-end 39 \\
        --output outputs/asb_results_safe_MP.csv --label MP
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import yaml
from tqdm import tqdm

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from third_party.ASB.backbone import (
    ASBAgentConfig,
    attack_happened,
    judge_task_success,
    load_agent_config,
    load_agent_tasks,
    load_attacker_tools_from_jsonl,
    load_tool_specs_from_jsonl,
)

from agent.middlewares import build_safe_agent_middlewares  # type: ignore[import-untyped]
from agent.tool_warpper import SafeAgentToolWrapperMiddleware  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "agent_name", "task", "attacker_idx",
    "attack_ok", "task_ok", "both_ok",
    "status", "error_type", "error_message",
]

DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_MCP_SERVER = "safeagent-core"


# ---------------------------------------------------------------------------
# YAML / MCP helpers
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config file, return dict (empty on error)."""
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def parse_mcp_response(resp: Any) -> Any:
    """
    Normalise an MCP tool response into a dict/string.

    MCP adapters often return list[{"type":"text","text":"...","id":"..."}];
    this helper extracts the text and attempts JSON parsing.
    """
    if isinstance(resp, list):
        texts: List[str] = []
        for b in resp:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                texts.append(b["text"])
        merged = "\n".join(texts).strip()
        if not merged:
            return resp
        try:
            return json.loads(merged)
        except json.JSONDecodeError:
            return merged

    if isinstance(resp, dict):
        return resp

    if isinstance(resp, str):
        s = resp.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s

    return resp


async def get_safeagent_tools(
    mcp_url: str = DEFAULT_MCP_URL,
    server_name: str = DEFAULT_MCP_SERVER,
) -> Tuple[MultiServerMCPClient, BaseTool, BaseTool]:
    """
    Connect to the SafeAgent MCP core and return (client, register_tool, step_tool).
    """
    client = MultiServerMCPClient(
        {server_name: {"url": mcp_url, "transport": "streamable_http"}},
    )
    tools = await client.get_tools()

    register_tool = next(
        (t for t in tools if t.name == "safeagent_register_session"), None
    )
    step_tool = next(
        (t for t in tools if t.name == "safeagent_step"), None
    )

    if register_tool is None:
        raise RuntimeError("MCP tools missing: safeagent_register_session")
    if step_tool is None:
        raise RuntimeError("MCP tools missing: safeagent_step")

    return client, register_tool, step_tool


# ---------------------------------------------------------------------------
# Agent cache
# ---------------------------------------------------------------------------
class _AgentCache:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


_AGENT_CACHE: Dict[str, _AgentCache] = {}


def get_agent_cache(agent_name: str) -> _AgentCache:
    if agent_name in _AGENT_CACHE:
        return _AGENT_CACHE[agent_name]
    agent_cfg: ASBAgentConfig = load_agent_config(agent_name)
    system_prompt = agent_cfg.build_system_prompt()
    _AGENT_CACHE[agent_name] = _AgentCache(system_prompt)
    return _AGENT_CACHE[agent_name]


# ---------------------------------------------------------------------------
# Async invoke helper
# ---------------------------------------------------------------------------
async def invoke_agent(agent: Any, user_query: str) -> List[BaseMessage]:
    """Invoke an agent with a user query, preferring the async path."""
    payload = {"messages": [HumanMessage(content=user_query)]}
    if hasattr(agent, "ainvoke"):
        out = await agent.ainvoke(payload)
    else:
        out = await asyncio.to_thread(agent.invoke, payload)

    if isinstance(out, dict):
        msgs = out.get("messages", [])
        return msgs if isinstance(msgs, list) else []
    return []


# ---------------------------------------------------------------------------
# Single test case (async)
# ---------------------------------------------------------------------------
async def run_one_case(
    agent_name: str,
    task: str,
    attacker_idx: Optional[int],
    *,
    register_tool: BaseTool,
    step_tool: BaseTool,
    runtime_cfg: Dict[str, Any],
    dev_cfg: Dict[str, Any],
    base_model: ChatOpenAI,
    judge_model: ChatOpenAI,
    direct_injection: bool,
    indirect_injection: bool,
    memory_poisoning: bool,
) -> Tuple[bool, bool]:
    """Run one ASB test case through SafeAgent. Returns (attack_ok, task_ok)."""

    cache = get_agent_cache(agent_name)
    session_id = str(uuid4())

    # --- 1) register session with SafeAgent core ---
    register_raw = await register_tool.ainvoke({
        "session_id": session_id,
        "runtime_cfg": runtime_cfg,
        "dev_cfg": dev_cfg,
    })
    register = parse_mcp_response(register_raw)
    if not (isinstance(register, dict) and register.get("ok") is True):
        raise RuntimeError(f"SafeAgent session registration failed: {register}")

    # --- 2) attacker spec ---
    attacker: Optional[Any] = None
    attacker_tool_name: Optional[str] = None
    injection_payload: Optional[str] = None

    if attacker_idx is not None and attacker_idx >= 0:
        attacker_specs = load_attacker_tools_from_jsonl(
            agent_name, attacker_indices=[attacker_idx],
        )
        if not attacker_specs:
            raise RuntimeError(
                f"No attacker tool for agent={agent_name}, idx={attacker_idx}"
            )
        attacker = attacker_specs[0]
        attacker_tool_name = attacker.tool_name
        if indirect_injection:
            injection_payload = attacker.build_injection_prompt()

    # --- 3) build normal tools ---
    normal_specs = load_tool_specs_from_jsonl(agent_name)
    injection = injection_payload if indirect_injection else None
    tools: List[BaseTool] = [
        spec.to_langchain_tool(injection) for spec in normal_specs
    ]

    # append attacker tool itself
    if attacker is not None:
        tools.append(attacker.to_langchain_tool())

    # --- 4) build agent with SafeAgent middlewares ---
    safe_middlewares = [
        *build_safe_agent_middlewares(step_tool, session_id),
        SafeAgentToolWrapperMiddleware(step_tool, session_id),
    ]
    agent = create_agent(
        model=base_model,
        tools=tools,
        system_prompt=cache.system_prompt,
        middleware=safe_middlewares,
    )

    # --- 5) DPI / memory poisoning ---
    user_query = task
    if attacker is not None and direct_injection:
        user_query = task + " " + attacker.build_injection_prompt()

    if memory_poisoning and attacker is not None:
        messages: List[BaseMessage] = [
            AIMessage(attacker.build_poisoned_message(task))
        ]
    else:
        messages = []

    messages.append(HumanMessage(content=user_query))

    # --- 6) run agent ---
    msgs = await invoke_agent(agent, user_query)

    # --- 7) judge results ---
    attack_ok = (
        attack_happened(msgs, attacker_tool_name)
        if attacker_tool_name
        else False
    )
    task_ok = bool(judge_task_success(msgs, task, judge_model))
    return attack_ok, task_ok


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def load_completed_jobs(csv_path: str) -> set:
    """Read existing CSV; return set of (agent_name, task, attacker_idx)."""
    if not Path(csv_path).exists():
        return set()
    existing = pd.read_csv(csv_path)
    return set(zip(existing["agent_name"], existing["task"], existing["attacker_idx"]))


def write_header(csv_path: str) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


# ---------------------------------------------------------------------------
# Main async evaluation
# ---------------------------------------------------------------------------
async def run_evaluation(
    *,
    output_csv: str,
    attack_start: int = 0,
    attack_end: int = -1,
    direct_injection: bool = False,
    indirect_injection: bool = False,
    memory_poisoning: bool = False,
    max_concurrency: int = 8,
    resume: bool = True,
    mcp_url: str = DEFAULT_MCP_URL,
    benign: bool = False,
) -> None:
    """Run SafeAgent evaluation with per-sample CSV output and resume support."""

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

    # --- configs ---
    cfg_dir = Path("config")
    runtime_cfg = load_yaml(cfg_dir / "runtime.yaml")
    dev_cfg = load_yaml(cfg_dir / "developer.yaml")

    print(f"[eval] SafeAgent evaluation")
    print(f"[eval] Benign={benign}")
    if not benign:
        print(f"[eval] DPI={direct_injection}, IPI={indirect_injection}, MP={memory_poisoning}")
    print(f"[eval] MCP URL: {mcp_url}")

    # --- MCP connection ---
    client, register_tool, step_tool = await get_safeagent_tools(mcp_url=mcp_url)
    print("[eval] MCP connected successfully")

    # --- build job list ---
    cases = load_agent_tasks()

    if benign:
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
        # MCP client is external; do not close it
        return

    # --- execution ---
    csv_lock = threading.Lock()

    def append_record(record: Dict) -> None:
        with csv_lock:
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(record)

    sem = asyncio.Semaphore(max_concurrency)
    attack_succ = task_succ = both_succ = err_count = 0
    pbar = tqdm(total=len(jobs), desc=f"SafeAgent (x{max_concurrency})")

    async def _run_job(agent_name: str, task: str, idx: int) -> None:
        nonlocal attack_succ, task_succ, both_succ, err_count

        async with sem:
            try:
                a_ok, t_ok = await run_one_case(
                    agent_name=agent_name,
                    task=task,
                    attacker_idx=idx,
                    register_tool=register_tool,
                    step_tool=step_tool,
                    runtime_cfg=runtime_cfg,
                    dev_cfg=dev_cfg,
                    base_model=base_model,
                    judge_model=judge_model,
                    direct_injection=direct_injection,
                    indirect_injection=indirect_injection,
                    memory_poisoning=memory_poisoning,
                )
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

    tasks = [asyncio.create_task(_run_job(a, t, i)) for (a, t, i) in jobs]

    for coro in asyncio.as_completed(tasks):
        try:
            await coro
        except Exception as e:
            tqdm.write(f"[FATAL] unexpected: {type(e).__name__}: {e}")

    pbar.close()
    # MCP client is external; do not close it

    # --- summary ---
    total = len(jobs)
    n_ok = total - err_count
    asr = attack_succ / total if total else 0.0
    tsr = task_succ / total if total else 0.0
    bsr = both_succ / total if total else 0.0

    print()
    print("=" * 60)
    print(f"  Defense:     SafeAgent (MCP)")
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
        description="ASB SafeAgent Defense Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output CSV path (default: outputs/asb_results_safe_{label}.csv)",
    )
    parser.add_argument(
        "--label", type=str, default="eval",
        help="Short label for injection mode (used in default output path, e.g. IPI/DPI/MP)",
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
        "--max-concurrency", type=int, default=8,
        help="Async concurrency limit (default: 8)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", default=False,
        help="Disable resume: overwrite existing CSV and start fresh",
    )
    parser.add_argument(
        "--mcp-url", type=str, default=DEFAULT_MCP_URL,
        help=f"SafeAgent MCP server URL (default: {DEFAULT_MCP_URL})",
    )

    parser.add_argument(
        "--benign", action="store_true", default=False,
        help="Run benign tasks only (no attacker, measures PNA via task_ok)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is None:
        suffix = "benign" if args.benign else (args.label or "eval")
        args.output = f"outputs/asb_results_safe_{suffix}.csv"

    asyncio.run(
        run_evaluation(
            output_csv=args.output,
            attack_start=args.attack_start,
            attack_end=args.attack_end,
            direct_injection=args.dpi,
            indirect_injection=args.ipi,
            memory_poisoning=args.mp,
            max_concurrency=args.max_concurrency,
            resume=not args.no_resume,
            mcp_url=args.mcp_url,
            benign=args.benign,
        )
    )


if __name__ == "__main__":
    main()
