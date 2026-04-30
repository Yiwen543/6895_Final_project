# Quantization Benchmark Design
**Project:** Nova Smart Home Assistant — EECS 6895 Final Project
**Date:** 2026-04-30
**Goal:** Compare GGUF quantization levels across model sizes to find the best accuracy/latency tradeoff for on-device inference on Raspberry Pi 5.

---

## 1. Scope

Add a standalone `benchmark_quantization.py` script that:
- Downloads missing GGUF model variants from HuggingFace (optional)
- Runs a fixed 20-case test suite against each variant
- Reports type accuracy, command accuracy, avg/P95 latency, and model size
- Saves results as `benchmark_results.csv` and `benchmark_results.md`

No changes to the existing Nova pipeline (`nova.py`, `agent.py`, `llm_parser.py`).

---

## 2. Model Variants

| Name | Path | Source | Est. Size |
|------|------|--------|-----------|
| 1.5B-Q3_K_M | `models/qwen2.5-1.5b-q3_k_m.gguf` | HF: Qwen/Qwen2.5-1.5B-Instruct-GGUF | ~0.8 GB |
| 1.5B-Q4_K_M | `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | already on Pi | ~1.1 GB |
| 1.5B-Q5_K_M | `models/qwen2.5-1.5b-q5_k_m.gguf` | HF: Qwen/Qwen2.5-1.5B-Instruct-GGUF | ~1.3 GB |
| 3B-Q3_K_M | `models/qwen2.5-3b-q3_k_m.gguf` | HF: Qwen/Qwen2.5-3B-Instruct-GGUF | ~1.5 GB |
| 3B-Q4_K_M | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | already on Pi (baseline) | ~2.0 GB |

Storage required: ~3.6 GB additional downloads; 23 GB free on Pi — no issue.

---

## 3. Test Suite (20 cases)

Each case: `{input, expected_type, expected_device, expected_action}`.
`expected_device` and `expected_action` are `None` for non-`direct_command` types.

| # | Input | Expected Type | Device | Action |
|---|-------|--------------|--------|--------|
| 1 | "Nova, turn on the light." | direct_command | light | turn_on |
| 2 | "Nova, turn off the light." | direct_command | light | turn_off |
| 3 | "Nova, set AC to 22 degrees." | direct_command | ac | set_temperature |
| 4 | "Nova, open the curtain." | direct_command | curtain | open |
| 5 | "Nova, set brightness to 70." | direct_command | light | set_brightness |
| 6 | "Nova, I feel cold." | needs_clarification | — | — |
| 7 | "Nova, it's a bit dark." | needs_clarification | — | — |
| 8 | "Nova, it's stuffy in here." | needs_clarification | — | — |
| 9 | "Nova, today's temperature is 28 degrees." | needs_clarification | — | — |
| 10 | "Nova, this room is boring." | needs_clarification | — | — |
| 11 | "Nova, what's your name?" | general_qa | — | — |
| 12 | "Nova, my name is Aston, what's your name?" | general_qa | — | — |
| 13 | "Nova, how do I eat an apple?" | general_qa | — | — |
| 14 | "Nova, what time is it?" | general_qa | — | — |
| 15 | "Nova, what's the weather like today?" | general_qa | — | — |
| 16 | "Hello." | invalid | — | — |
| 17 | "Never mind." | invalid | — | — |
| 18 | "Um." | invalid | — | — |
| 19 | "Okay." | invalid | — | — |
| 20 | "Nova." | invalid | — | — |

---

## 4. Evaluation Metrics

Per model variant, computed over all 20 cases × 3 repetitions (median per case):

| Metric | Definition |
|--------|-----------|
| **Type accuracy** | % of cases where `type` field exactly matches expected |
| **Command accuracy** | % of `direct_command` cases where both `device` and `action` match |
| **Avg latency (ms)** | Mean of per-case median latencies |
| **P95 latency (ms)** | 95th percentile of per-case median latencies |
| **Model size (GB)** | GGUF file size on disk |

Each case is run 3 times; median latency is used to suppress cold-start variance. The same `UNIFIED_SYSTEM_PROMPT`, `n_threads=4`, `n_ctx=768`, `temperature=0`, `stop=["\n","Input:"]` settings as production are used for all variants.

---

## 5. Script Architecture

```
benchmark_quantization.py
├── MODEL_VARIANTS: list[dict]     # name, path, hf_url, hf_filename, size_gb
├── TEST_CASES: list[dict]         # input, expected_type, device, action
├── SYSTEM_PROMPT                  # imported from llm_parser.UNIFIED_SYSTEM_PROMPT
├── download_missing_models()      # wget if not present, skips if exists
├── run_one_case(llm, case) -> (predicted_type, device, action, latency_ms)
│     # runs 3x, returns median latency + majority-vote prediction
├── run_model(variant) -> list[CaseResult]
│     # loads Llama, iterates test cases, unloads
├── evaluate(results) -> ModelMetrics
│     # computes type_acc, cmd_acc, avg_ms, p95_ms
├── print_table(all_metrics)       # aligned console table
└── save_outputs(all_metrics)      # writes .csv and .md
```

**CLI:**
```bash
python3 benchmark_quantization.py              # only tests already-downloaded models
python3 benchmark_quantization.py --download   # downloads missing variants first
python3 benchmark_quantization.py --models 1.5B-Q4_K_M,3B-Q4_K_M  # subset
```

---

## 6. Output Files

**`benchmark_results.md`** — Markdown table for copy-paste into report:
```
| Model       | Type Acc | Cmd Acc | Avg (ms) | P95 (ms) | Size (GB) |
|-------------|----------|---------|----------|----------|-----------|
| 1.5B-Q3_K_M | ...      | ...     | ...      | ...      | 0.8       |
...
```

**`benchmark_results.csv`** — raw data for plotting:
```
model,type_acc,cmd_acc,avg_ms,p95_ms,size_gb
1.5B-Q3_K_M,...
```

---

## 7. Success Criteria

- Script runs to completion on Pi 5 without crashing
- Baseline (3B-Q4_K_M) matches manually observed ~90% type accuracy
- Results table is copy-pasteable into the EECS 6895 final report evaluation section
