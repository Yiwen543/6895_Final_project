"""Unit tests for benchmark_quantization helper functions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from benchmark_quantization import evaluate, format_row


# ── evaluate() ────────────────────────────────────────────────────────────────

def _make_results(overrides: list) -> list:
    base = {
        "expected_type": "direct_command",
        "predicted_type": "direct_command",
        "expected_device": "light",
        "predicted_device": "light",
        "expected_action": "turn_on",
        "predicted_action": "turn_on",
        "latency_ms": 1000.0,
    }
    return [{**base, **o} for o in overrides]


def test_evaluate_perfect_accuracy():
    results = _make_results([{} for _ in range(5)])
    metrics = evaluate(results)
    assert metrics["type_acc"] == 1.0
    assert metrics["cmd_acc"] == 1.0


def test_evaluate_type_mismatch():
    results = _make_results([
        {},
        {"predicted_type": "invalid"},
    ])
    metrics = evaluate(results)
    assert metrics["type_acc"] == pytest.approx(0.5)


def test_evaluate_cmd_acc_only_counts_direct_command():
    results = _make_results([
        {"expected_type": "direct_command", "predicted_type": "direct_command",
         "predicted_device": "light", "predicted_action": "turn_on"},
        {"expected_type": "needs_clarification", "predicted_type": "needs_clarification",
         "expected_device": None, "predicted_device": None,
         "expected_action": None, "predicted_action": None},
    ])
    metrics = evaluate(results)
    assert metrics["type_acc"] == 1.0
    assert metrics["cmd_acc"] == 1.0


def test_evaluate_cmd_acc_wrong_device():
    results = _make_results([
        {"predicted_device": "ac"},
    ])
    metrics = evaluate(results)
    assert metrics["cmd_acc"] == 0.0


def test_evaluate_cmd_acc_wrong_action():
    results = _make_results([
        {"predicted_action": "turn_off"},   # wrong action
    ])
    metrics = evaluate(results)
    assert metrics["cmd_acc"] == 0.0


def test_evaluate_latency():
    results = _make_results([
        {"latency_ms": 1000.0},
        {"latency_ms": 2000.0},
        {"latency_ms": 3000.0},
        {"latency_ms": 4000.0},
    ])
    metrics = evaluate(results)
    assert metrics["avg_ms"] == pytest.approx(2500.0)
    assert metrics["p95_ms"] >= 3000.0


# ── format_row() ──────────────────────────────────────────────────────────────

def test_format_row_basic():
    row = format_row("1.5B-Q4_K_M", {"type_acc": 0.85, "cmd_acc": 1.0,
                                       "avg_ms": 5800.0, "p95_ms": 6900.0}, 1.1)
    assert "1.5B-Q4_K_M" in row
    assert "85%" in row
    assert "100%" in row
    assert "5800" in row
    assert "1.1" in row
