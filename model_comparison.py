"""
model_comparison.py — Nova Smart Home Assistant: 三模型横向对比评测脚本

加载 TinyLlama-1.1B-Chat、Qwen2.5-1.5B-Instruct、Phi-2 三个模型，
使用相同的测试用例进行推理，对比分类准确率与推理延迟，
并将结果输出到终端 + 生成 comparison_log.txt 日志文件。

用法:
    python model_comparison.py
"""

import os
import re
import json
import time
import datetime
import torch
from typing import Dict, Any, Optional, List

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ─────────────────────────────────────────────
# 1. 配置
# ─────────────────────────────────────────────

MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen2.5-1.5B":   "Qwen/Qwen2.5-1.5B-Instruct",
    "Phi-2-2.7B":     "microsoft/phi-2",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# ─────────────────────────────────────────────
# 2. 统一 System Prompt（与 Nova_4_16.ipynb Section 5 一致）
# ─────────────────────────────────────────────

UNIFIED_SYSTEM_PROMPT = """
Return exactly one JSON object and nothing else.

Allowed outputs:

{"type":"direct_command","device":"light|curtain|window|ac","action":"turn_on|turn_off|set_brightness|rgb_cycle|open|close|set_position|set_temperature","value":null_or_int}

{"type":"needs_clarification","question":"...","options":["...","..."]}

{"type":"general_qa","answer":"..."}

{"type":"invalid"}

CRITICAL classification rules:
- direct_command: ONLY when the user explicitly names a device AND a specific action. Example: "turn on the light", "set AC to 24", "open the curtain".
- needs_clarification: When the user describes a feeling, comfort, mood, or indirect desire WITHOUT naming a specific action. This includes: feeling cold, feeling hot, it's dark, it's bright, room is boring, complaining about a device, wanting a vibe change. NEVER output direct_command for these.
- general_qa: When the user asks a question unrelated to home device control. Example: "how to eat an apple", "what time is it".
- invalid: When there is no meaningful request.

Examples:

Input: Nova, turn on the light.
Output: {"type":"direct_command","device":"light","action":"turn_on","value":null}

Input: Nova, turn off the light.
Output: {"type":"direct_command","device":"light","action":"turn_off","value":null}

Input: Nova, set the AC to 24 degrees.
Output: {"type":"direct_command","device":"ac","action":"set_temperature","value":24}

Input: Nova, open the curtain.
Output: {"type":"direct_command","device":"curtain","action":"open","value":null}

Input: Nova, close the window.
Output: {"type":"direct_command","device":"window","action":"close","value":null}

Input: Nova, I feel cold.
Output: {"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"]}

Input: Nova, I feel a little bit cold.
Output: {"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"]}

Input: Nova, it's a bit dark.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"]}

Input: Nova, this room is too dark.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"]}

Input: Nova, I feel hot.
Output: {"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"]}

Input: Nova, fuck this light.
Output: {"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"]}

Input: Nova, this light is annoying.
Output: {"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"]}

Input: Nova, make this room lively.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"]}

Input: Nova, how do I eat an apple?
Output: {"type":"general_qa","answer":"Wash it first, then eat it."}

Input: Nova, can I still eat this dish after a night in the fridge?
Output: {"type":"general_qa","answer":"Yes, most cooked food is safe for up to 3-4 days in the fridge."}

Input: Hello.
Output: {"type":"invalid"}
""".strip()

# ─────────────────────────────────────────────
# 3. 测试用例
# ─────────────────────────────────────────────

TEST_CASES = [
    # ---- direct_command ----
    {
        "input": "Nova, turn on the light.",
        "expected_type": "direct_command",
        "description": "明确设备+动作：开灯",
    },
    {
        "input": "Nova, turn off the light.",
        "expected_type": "direct_command",
        "description": "明确设备+动作：关灯",
    },
    {
        "input": "Nova, set the AC to 24 degrees.",
        "expected_type": "direct_command",
        "description": "明确设备+数值：空调设温度",
    },
    {
        "input": "Nova, open the curtain.",
        "expected_type": "direct_command",
        "description": "明确设备+动作：开窗帘",
    },
    {
        "input": "Nova, close the window.",
        "expected_type": "direct_command",
        "description": "明确设备+动作：关窗户",
    },

    # ---- needs_clarification ----
    {
        "input": "Nova, I feel a little bit cold.",
        "expected_type": "needs_clarification",
        "description": "间接需求：感觉冷（不指定设备）",
    },
    {
        "input": "Nova, it's a bit dark.",
        "expected_type": "needs_clarification",
        "description": "间接需求：有点暗（不指定动作）",
    },
    {
        "input": "Nova, fuck this light.",
        "expected_type": "needs_clarification",
        "description": "间接需求：抱怨灯光（无明确动作）",
    },
    {
        "input": "Nova, make this room lively.",
        "expected_type": "needs_clarification",
        "description": "间接需求：氛围请求",
    },
    {
        "input": "Nova, I feel hot.",
        "expected_type": "needs_clarification",
        "description": "间接需求：感觉热",
    },
    {
        "input": "Nova, this light is annoying.",
        "expected_type": "needs_clarification",
        "description": "间接需求：抱怨灯光",
    },

    # ---- general_qa ----
    {
        "input": "Nova, how can I eat an apple?",
        "expected_type": "general_qa",
        "description": "通用问答：与设备无关的问题",
    },
    {
        "input": "Nova, can I still eat this dish after one night in the fridge?",
        "expected_type": "general_qa",
        "description": "通用问答：食物保存问题",
    },
    {
        "input": "Nova, what time is it?",
        "expected_type": "general_qa",
        "description": "通用问答：询问时间",
    },

    # ---- invalid ----
    {
        "input": "Hello.",
        "expected_type": "invalid",
        "description": "无效输入：无唤醒词、无请求",
    },
    {
        "input": "Never mind.",
        "expected_type": "invalid",
        "description": "无效输入：取消请求",
    },
]

# ─────────────────────────────────────────────
# 4. JSON 解析工具函数
# ─────────────────────────────────────────────

def extract_first_json_object(text: str) -> Optional[str]:
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
                return text[start : i + 1]
    return None


def has_complete_json_object(text: str) -> bool:
    start = text.find("{")
    if start == -1:
        return False
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


def normalize_unified_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"type": "invalid"}

    result_type = obj.get("type", "invalid")
    if not isinstance(result_type, str):
        return {"type": "invalid"}

    result_type = result_type.strip().lower()

    if result_type == "direct_command":
        device_v = obj.get("device")
        action_v = obj.get("action")
        value_v = obj.get("value")
        if not isinstance(device_v, str) or not isinstance(action_v, str):
            return {"type": "invalid"}
        device_v = device_v.strip().lower()
        action_v = action_v.strip().lower()
        if "|" in device_v or "|" in action_v:
            return {"type": "invalid"}
        if value_v in ["null", "None", "none", "NULL"]:
            value_v = None
        if isinstance(value_v, str):
            value_v = value_v.strip().replace("%", "")
            if re.fullmatch(r"\d{1,3}", value_v):
                value_v = int(value_v)
        return {"type": "direct_command", "device": device_v, "action": action_v, "value": value_v}

    if result_type == "needs_clarification":
        question = obj.get("question", "")
        options = obj.get("options", [])
        if not isinstance(question, str) or not isinstance(options, list):
            return {"type": "invalid"}
        options = [str(x).strip() for x in options if str(x).strip()]
        if len(options) == 0:
            return {"type": "invalid"}
        return {"type": "needs_clarification", "question": question.strip(), "options": options}

    if result_type == "general_qa":
        answer = obj.get("answer", "")
        if not isinstance(answer, str):
            return {"type": "invalid"}
        return {"type": "general_qa", "answer": answer.strip()}

    return {"type": "invalid"}


# ─────────────────────────────────────────────
# 5. 停止策略：JSON 对象完成时停止生成
# ─────────────────────────────────────────────

class JsonStopOnComplete(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_length :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return has_complete_json_object(text)


# ─────────────────────────────────────────────
# 6. 推理函数
# ─────────────────────────────────────────────

def build_prompt(tokenizer, model_name: str, system_prompt: str, user_text: str) -> str:
    """
    根据不同模型构建 chat prompt。
    Phi-2 没有 chat_template，使用 Instruct 格式手动拼接。
    """
    if "phi-2" in model_name.lower():
        # Phi-2 不支持 apply_chat_template，使用简单 Instruct 格式
        return (
            f"Instruct: {system_prompt}\n\n"
            f"User: {user_text}\n"
            f"Output:"
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def run_inference(model, tokenizer, model_name: str, user_text: str, max_new_tokens: int = 96):
    """
    对单条输入执行推理，返回 (normalized_result, raw_output, latency_ms)。
    """
    prompt_text = build_prompt(
        tokenizer, model_name, UNIFIED_SYSTEM_PROMPT,
        f'Text: "{user_text}"\nReturn JSON only.'
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList([
        JsonStopOnComplete(tokenizer, prompt_length)
    ])

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    generated_ids = outputs[0][prompt_length:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    json_str = extract_first_json_object(raw_output)
    if json_str is None:
        return {"type": "invalid"}, raw_output, latency_ms

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return {"type": "invalid"}, raw_output, latency_ms

    return normalize_unified_result(parsed), raw_output, latency_ms


# ─────────────────────────────────────────────
# 7. 评测主流程
# ─────────────────────────────────────────────

def evaluate_model(model, tokenizer, model_name: str, test_cases: List[dict]) -> List[dict]:
    """对一个模型运行所有测试用例，返回结果列表。"""
    results = []
    for tc in test_cases:
        user_input = tc["input"]
        expected_type = tc["expected_type"]

        semantic, raw_output, latency_ms = run_inference(
            model, tokenizer, model_name, user_input
        )

        predicted_type = semantic.get("type", "invalid")
        correct = (predicted_type == expected_type)

        results.append({
            "input":          user_input,
            "description":    tc["description"],
            "expected_type":  expected_type,
            "predicted_type": predicted_type,
            "correct":        correct,
            "latency_ms":     round(latency_ms, 1),
            "raw_output":     raw_output,
            "parsed":         semantic,
        })

    return results


def print_section(title: str, char: str = "=", width: int = 90):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def format_results_table(results: List[dict]) -> str:
    """将单个模型的结果格式化为文本表格。"""
    lines = []
    header = f"{'#':<3} {'Input':<50} {'Expected':<22} {'Predicted':<22} {'Match':<6} {'Latency':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, r in enumerate(results, 1):
        inp = r["input"][:48] + ".." if len(r["input"]) > 50 else r["input"]
        mark = "OK" if r["correct"] else "FAIL"
        lines.append(
            f"{i:<3} {inp:<50} {r['expected_type']:<22} {r['predicted_type']:<22} {mark:<6} {r['latency_ms']:>8.1f} ms"
        )

    return "\n".join(lines)


def format_summary(all_results: Dict[str, List[dict]]) -> str:
    """生成三模型汇总对比表。"""
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("  SUMMARY — Model Comparison")
    lines.append("=" * 90)

    header = f"{'Metric':<35} ",
    model_names = list(all_results.keys())
    header_line = f"{'Metric':<35}"
    for name in model_names:
        header_line += f" {name:>18}"
    lines.append(header_line)
    lines.append("-" * (35 + 19 * len(model_names)))

    # 总准确率
    acc_line = f"{'Overall Accuracy':<35}"
    for name in model_names:
        results = all_results[name]
        acc = sum(1 for r in results if r["correct"]) / len(results)
        acc_line += f" {acc:>17.1%}"
    lines.append(acc_line)

    # 分类别准确率
    categories = ["direct_command", "needs_clarification", "general_qa", "invalid"]
    for cat in categories:
        cat_line = f"  ↳ {cat:<31}"
        for name in model_names:
            results = all_results[name]
            cat_results = [r for r in results if r["expected_type"] == cat]
            if cat_results:
                cat_acc = sum(1 for r in cat_results if r["correct"]) / len(cat_results)
                cat_line += f" {cat_acc:>17.1%}"
            else:
                cat_line += f" {'N/A':>17}"
        lines.append(cat_line)

    # 平均延迟
    lat_line = f"{'Avg Latency (ms)':<35}"
    for name in model_names:
        results = all_results[name]
        avg_lat = sum(r["latency_ms"] for r in results) / len(results)
        lat_line += f" {avg_lat:>15.1f} ms"
    lines.append(lat_line)

    # 最大延迟
    max_line = f"{'Max Latency (ms)':<35}"
    for name in model_names:
        results = all_results[name]
        max_lat = max(r["latency_ms"] for r in results)
        max_line += f" {max_lat:>15.1f} ms"
    lines.append(max_line)

    # 最小延迟
    min_line = f"{'Min Latency (ms)':<35}"
    for name in model_names:
        results = all_results[name]
        min_lat = min(r["latency_ms"] for r in results)
        min_line += f" {min_lat:>15.1f} ms"
    lines.append(min_line)

    lines.append("")

    # 逐项错误汇总
    lines.append("-" * 90)
    lines.append("  Misclassified Cases")
    lines.append("-" * 90)

    for name in model_names:
        failures = [r for r in all_results[name] if not r["correct"]]
        if failures:
            lines.append(f"\n  [{name}] — {len(failures)} error(s):")
            for f in failures:
                lines.append(
                    f"    • \"{f['input'][:60]}\" → expected: {f['expected_type']}, got: {f['predicted_type']}"
                )
        else:
            lines.append(f"\n  [{name}] — All correct!")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(os.path.dirname(__file__), f"comparison_log_{timestamp}.txt")

    log_lines = []

    def log(msg: str = ""):
        print(msg)
        log_lines.append(msg)

    log(f"Nova Model Comparison — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: {DEVICE}  |  dtype: {DTYPE}")
    log(f"Test cases: {len(TEST_CASES)}")
    log(f"Models: {', '.join(MODELS.keys())}")

    all_results: Dict[str, List[dict]] = {}

    for model_label, model_id in MODELS.items():
        print_section(f"Loading model: {model_label}  ({model_id})")
        log(f"\n{'='*90}")
        log(f"  Loading model: {model_label}  ({model_id})")
        log(f"{'='*90}")

        try:
            load_start = time.perf_counter()

            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=DTYPE,
                device_map="auto",
                trust_remote_code=True,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            load_sec = time.perf_counter() - load_start
            log(f"  Model loaded in {load_sec:.1f}s")
            log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        except Exception as e:
            log(f"  [ERROR] Failed to load {model_label}: {e}")
            log(f"  Skipping this model.\n")
            continue

        # 运行评测
        log(f"\n  Running {len(TEST_CASES)} test cases...")
        results = evaluate_model(model, tokenizer, model_label, TEST_CASES)
        all_results[model_label] = results

        # 打印该模型的详细结果
        table = format_results_table(results)
        log(f"\n{table}")

        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        log(f"\n  Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
        log(f"  Avg Latency: {avg_latency:.1f} ms")

        # 释放显存/内存
        log(f"\n  Unloading {model_label}...")
        del model
        del tokenizer
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        import gc; gc.collect()
        log(f"  Done.")

    # 汇总对比
    if len(all_results) > 1:
        summary = format_summary(all_results)
        log(summary)

    # 写入日志文件
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n{'='*90}")
    print(f"  Log saved to: {log_path}")
    print(f"{'='*90}")

    # 同时保存 JSON 格式的详细结果
    json_path = log_path.replace(".txt", ".json")
    json_data = {}
    for model_name, results in all_results.items():
        json_data[model_name] = {
            "accuracy": sum(1 for r in results if r["correct"]) / len(results),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1),
            "results": [
                {
                    "input": r["input"],
                    "expected_type": r["expected_type"],
                    "predicted_type": r["predicted_type"],
                    "correct": r["correct"],
                    "latency_ms": r["latency_ms"],
                    "raw_output": r["raw_output"],
                }
                for r in results
            ],
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"  JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
