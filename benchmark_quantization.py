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
    last_parsed = {"type": "invalid", "device": None, "action": None}

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
        last_parsed = parse_output(raw)

    median_ms = statistics.median(latencies)
    return {
        "input":            case["input"],
        "expected_type":    case["type"],
        "predicted_type":   last_parsed["type"],
        "expected_device":  case["device"],
        "predicted_device": last_parsed["device"],
        "expected_action":  case["action"],
        "predicted_action": last_parsed["action"],
        "latency_ms":       median_ms,
    }
