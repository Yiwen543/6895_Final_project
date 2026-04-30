#!/usr/bin/env python3
"""
Quantization benchmark for Nova LLM — EECS 6895 Final Project.

Runs a fixed 20-case intent-classification test suite across multiple
Qwen2.5 GGUF variants and reports accuracy + latency metrics.

Usage:
    python3 benchmark_quantization.py                            # only already-downloaded models
    python3 benchmark_quantization.py --download                 # fetch missing variants first
    python3 benchmark_quantization.py --models 1.5B-Q4_K_M,3B-Q4_K_M
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Inference settings (match production) ────────────────────────────────────

N_CTX     = 768   # covers SYSTEM_PROMPT (~430 tok) + longest test input + reply headroom
N_THREADS = 4
MAX_TOKENS = 60
STOP_SEQS  = ["\n", "Input:"]
TEMPERATURE = 0.0
REPEAT_PENALTY = 1.1
REPETITIONS = 3   # runs per test case; median latency is used

SYSTEM_PROMPT = """
Return exactly one JSON object and nothing else.

Outputs:
{"type":"direct_command","device":"light|curtain|window|ac","action":"turn_on|turn_off|set_brightness|rgb_cycle|open|close|set_position|set_temperature","value":null_or_int,"reply":"brief confirmation"}
{"type":"needs_clarification","question":"...","options":["...","..."],"reply":"question restated for speech"}
{"type":"general_qa","answer":"..."}
{"type":"invalid"}

Rules:
- direct_command: user explicitly names BOTH a device AND a specific action. reply: short friendly confirmation.
- needs_clarification: user describes a feeling, discomfort, or vague atmosphere about the home (cold, hot, dark, boring, annoying device) WITHOUT naming a specific action. reply: the clarification question.
- general_qa: any question unrelated to controlling home devices (food, health, science, time, etc.).
- invalid: no meaningful request.

Examples:
Input: Nova, turn on the light.
Output: {"type":"direct_command","device":"light","action":"turn_on","value":null,"reply":"Sure, turning on the light!"}

Input: Nova, I feel cold.
Output: {"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}

Input: Nova, it's a bit dark.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}

Input: Nova, how do I eat an apple?
Output: {"type":"general_qa","answer":"Wash it first, then eat it."}

Input: Hello.
Output: {"type":"invalid"}
""".strip()

# ── Model registry ────────────────────────────────────────────────────────────

MODEL_VARIANTS: List[Dict[str, Any]] = [
    {
        "name":     "1.5B-Q3_K_M",
        "path":     "models/qwen2.5-1.5b-instruct-q3_k_m.gguf",
        "url":      "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q3_k_m.gguf",
        "size_gb":  0.8,
    },
    {
        "name":     "1.5B-Q4_K_M",
        "path":     "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url":      None,   # pre-downloaded on Pi; --download skips this
        "size_gb":  1.1,
    },
    {
        "name":     "1.5B-Q5_K_M",
        "path":     "models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
        "url":      "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q5_k_m.gguf",
        "size_gb":  1.3,
    },
    {
        "name":     "3B-Q3_K_M",
        "path":     "models/qwen2.5-3b-instruct-q3_k_m.gguf",
        "url":      "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q3_k_m.gguf",
        "size_gb":  1.5,
    },
    {
        "name":     "3B-Q4_K_M",
        "path":     "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "url":      None,   # pre-downloaded on Pi; --download skips this
        "size_gb":  2.0,
    },
]

# ── Test suite ────────────────────────────────────────────────────────────────

TEST_CASES: List[Dict[str, Any]] = [
    # direct_command (5)
    {"input": "Nova, turn on the light.",    "type": "direct_command", "device": "light",   "action": "turn_on"},
    {"input": "Nova, turn off the light.",   "type": "direct_command", "device": "light",   "action": "turn_off"},
    {"input": "Nova, set AC to 22 degrees.", "type": "direct_command", "device": "ac",      "action": "set_temperature"},
    {"input": "Nova, open the curtain.",     "type": "direct_command", "device": "curtain", "action": "open"},
    {"input": "Nova, set brightness to 70.", "type": "direct_command", "device": "light",   "action": "set_brightness"},
    # needs_clarification (5)
    {"input": "Nova, I feel cold.",                       "type": "needs_clarification", "device": None, "action": None},
    {"input": "Nova, it's a bit dark.",                   "type": "needs_clarification", "device": None, "action": None},
    {"input": "Nova, it's stuffy in here.",               "type": "needs_clarification", "device": None, "action": None},
    {"input": "Nova, today's temperature is 28 degrees.", "type": "needs_clarification", "device": None, "action": None},
    {"input": "Nova, this room is boring.",               "type": "needs_clarification", "device": None, "action": None},
    # general_qa (5)
    {"input": "Nova, what's your name?",                          "type": "general_qa", "device": None, "action": None},
    {"input": "Nova, my name is Aston, what's your name?",        "type": "general_qa", "device": None, "action": None},
    {"input": "Nova, how do I eat an apple?",                     "type": "general_qa", "device": None, "action": None},
    {"input": "Nova, what time is it?",                           "type": "general_qa", "device": None, "action": None},
    {"input": "Nova, what's the weather like today?",             "type": "general_qa", "device": None, "action": None},
    # invalid (5)
    {"input": "Hello.",       "type": "invalid", "device": None, "action": None},
    {"input": "Never mind.",  "type": "invalid", "device": None, "action": None},
    {"input": "Um.",          "type": "invalid", "device": None, "action": None},
    {"input": "Okay.",        "type": "invalid", "device": None, "action": None},
    {"input": "Nova.",        "type": "invalid", "device": None, "action": None},
]

# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate(raw_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute metrics from a list of per-case result dicts.

    Each dict must have: expected_type, predicted_type, expected_device,
    predicted_device, expected_action, predicted_action, latency_ms.
    """
    type_correct = 0
    cmd_total = 0
    cmd_correct = 0
    latencies = []

    for r in raw_results:
        if r["predicted_type"] == r["expected_type"]:
            type_correct += 1
        if r["expected_type"] == "direct_command":
            cmd_total += 1
            if (r["predicted_device"] == r["expected_device"] and
                    r["predicted_action"] == r["expected_action"]):
                cmd_correct += 1
        latencies.append(r["latency_ms"])

    n = len(raw_results)
    latencies_sorted = sorted(latencies)
    p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)

    return {
        "type_acc": type_correct / n if n else 0.0,
        "cmd_acc":  cmd_correct / cmd_total if cmd_total else 0.0,
        "avg_ms":   statistics.mean(latencies) if latencies else 0.0,
        "p95_ms":   latencies_sorted[p95_idx] if latencies_sorted else 0.0,
    }


def format_row(name: str, metrics: Dict[str, float], size_gb: float) -> str:
    """Return a Markdown table row string."""
    return (
        f"| {name:<12} "
        f"| {metrics['type_acc']*100:>6.0f}% "
        f"| {metrics['cmd_acc']*100:>6.0f}% "
        f"| {metrics['avg_ms']:>8.0f} "
        f"| {metrics['p95_ms']:>8.0f} "
        f"| {size_gb:>8.1f} |"
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def _extract_first_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_output(raw: str) -> Dict[str, Any]:
    """Extract type/device/action from raw LLM output string."""
    json_str = _extract_first_json(raw)
    if not json_str:
        return {"type": "invalid", "device": None, "action": None}
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return {"type": "invalid", "device": None, "action": None}
    t = str(obj.get("type", "invalid")).strip().lower()
    if t not in ("direct_command", "needs_clarification", "general_qa", "invalid"):
        t = "invalid"
    device = str(obj.get("device", "")).strip().lower() or None
    action = str(obj.get("action", "")).strip().lower() or None
    if t != "direct_command":
        device = None
        action = None
    return {"type": t, "device": device, "action": action}


def run_one_case(llm, case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case REPETITIONS times; return result with median latency."""
    user_prompt = f'Text: "{case["input"]}"\nReturn JSON only.'
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    latencies = []
    all_parsed = []

    for _ in range(REPETITIONS):
        t0 = time.perf_counter()
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            repeat_penalty=REPEAT_PENALTY,
            stop=STOP_SEQS,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        raw = resp["choices"][0]["message"]["content"].strip()
        all_parsed.append(parse_output(raw))

    # majority vote on type; use the first result with the winning type for device/action
    from collections import Counter
    type_counts = Counter(p["type"] for p in all_parsed)
    winning_type = type_counts.most_common(1)[0][0]
    winner = next(p for p in all_parsed if p["type"] == winning_type)

    median_ms = statistics.median(latencies)
    return {
        "input":            case["input"],
        "expected_type":    case["type"],
        "predicted_type":   winner["type"],
        "expected_device":  case["device"],
        "predicted_device": winner["device"],
        "expected_action":  case["action"],
        "predicted_action": winner["action"],
        "latency_ms":       median_ms,
    }


def run_model(variant: Dict[str, Any], subset: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load one GGUF variant, run all test cases, return raw results."""
    from llama_cpp import Llama

    path = variant["path"]
    if not Path(path).exists():
        print(f"  SKIP — file not found: {path}")
        return []

    print(f"\n{'='*60}")
    print(f"  Model: {variant['name']}  ({path})")
    print(f"{'='*60}")

    llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)

    cases = TEST_CASES
    if subset:
        cases = [c for c in TEST_CASES if c["type"] in subset]

    results = []
    for i, case in enumerate(cases, 1):
        result = run_one_case(llm, case)
        correct = "✓" if result["predicted_type"] == result["expected_type"] else "✗"
        print(f"  [{i:02d}/{len(cases)}] {correct} {result['latency_ms']:>7.0f}ms  "
              f"{result['expected_type']:<22} → {result['predicted_type']}")
        results.append(result)

    del llm
    return results


def download_missing_models(variants: List[Dict[str, Any]]) -> None:
    """Download GGUF files that don't exist locally, skip if already present."""
    for v in variants:
        if Path(v["path"]).exists():
            print(f"  Already exists: {v['path']}")
            continue
        if not v["url"]:
            print(f"  No URL for {v['name']} — skipping")
            continue
        print(f"\nDownloading {v['name']} (~{v['size_gb']} GB) ...")
        Path(v["path"]).parent.mkdir(parents=True, exist_ok=True)
        ret = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", v["path"], v["url"]],
            check=False,
        )
        if ret.returncode != 0:
            print(f"  ERROR: download failed for {v['name']}")
            Path(v["path"]).unlink(missing_ok=True)


_TABLE_HEADER = (
    "| Model        | Type Acc | Cmd Acc | Avg (ms) | P95 (ms) | Size (GB) |\n"
    "|--------------|----------|---------|----------|----------|-----------|"
)


def print_table(all_metrics: List[Tuple[str, Dict[str, float], float]]) -> None:
    """Print aligned comparison table to stdout."""
    print(f"\n{_TABLE_HEADER}")
    for name, metrics, size_gb in all_metrics:
        print(format_row(name, metrics, size_gb))


def save_outputs(all_metrics: List[Tuple[str, Dict[str, float], float]],
                 all_raw: List[Tuple[str, List[Dict[str, Any]]]]) -> None:
    """Write benchmark_results.csv and benchmark_results.md."""
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "type_acc", "cmd_acc",
                                                "avg_ms", "p95_ms", "size_gb"])
        writer.writeheader()
        for name, metrics, size_gb in all_metrics:
            writer.writerow({
                "model":    name,
                "type_acc": round(metrics["type_acc"], 4),
                "cmd_acc":  round(metrics["cmd_acc"], 4),
                "avg_ms":   round(metrics["avg_ms"], 1),
                "p95_ms":   round(metrics["p95_ms"], 1),
                "size_gb":  size_gb,
            })
    print("\nSaved: benchmark_results.csv")

    with open("benchmark_results.md", "w") as f:
        f.write("# Nova LLM Quantization Benchmark\n\n")
        f.write(_TABLE_HEADER + "\n")
        for name, metrics, size_gb in all_metrics:
            f.write(format_row(name, metrics, size_gb) + "\n")
        f.write("\n_Metrics: Type Acc = correct intent type / 20 cases; "
                "Cmd Acc = correct device+action / 5 direct_command cases; "
                f"Avg/P95 latency in ms ({REPETITIONS} runs median per case)._\n")
    print("Saved: benchmark_results.md")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Nova LLM quantization benchmark")
    parser.add_argument("--download", action="store_true",
                        help="Download missing GGUF variants before benchmarking")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to test, e.g. 1.5B-Q4_K_M,3B-Q4_K_M")
    args = parser.parse_args()

    variants = MODEL_VARIANTS
    if args.models:
        names = {n.strip() for n in args.models.split(",")}
        variants = [v for v in MODEL_VARIANTS if v["name"] in names]
        if not variants:
            print(f"No matching models found. Available: {[v['name'] for v in MODEL_VARIANTS]}")
            sys.exit(1)

    if args.download:
        print("==> Checking / downloading missing models ...")
        download_missing_models(variants)

    all_metrics = []
    all_raw = []
    for v in variants:
        raw = run_model(v)
        if not raw:
            continue
        metrics = evaluate(raw)
        all_metrics.append((v["name"], metrics, v["size_gb"]))
        all_raw.append((v["name"], raw))

    if not all_metrics:
        print("\nNo models were benchmarked. Use --download to fetch missing variants.")
        sys.exit(1)

    print_table(all_metrics)
    save_outputs(all_metrics, all_raw)


if __name__ == "__main__":
    main()
