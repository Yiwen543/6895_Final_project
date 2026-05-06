# Cathey 软件端修复设计：TinyLlama + LoRA + Bug Fix

**日期**：2026-05-06  
**分支**：embedded  
**状态**：已批准

---

## 目标

1. 将部署 LLM 从 Qwen2.5-3B 切换到 TinyLlama-1.1B + LoRA adapter
2. 修复 `Hello.` → `invalid` 分类错误（应为 `general_qa`）
3. 修复 general_qa 记忆召回失效（`parse_unified` context 不含 prefs，`pre_answer` 绕过 `answer_qa`）
4. 补充 `lower the temperature` → `needs_clarification` 的 prompt 示例与训练样本

---

## 改动范围

### `config.py`

- `LLM_MODEL_NAME` 改为 `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"`
- 新增常量 `LORA_ADAPTER_PATH = "cathey_lora_adapter/final_adapter"`
- `LLM_GGUF_PATH` 保留不动（Pi GGUF 路径，后续合并 adapter 后另行处理）

### `llm_parser.py`

**模型加载**（`transformers` backend）：
- 加载 base model 后，调 `PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)` 挂载 LoRA adapter
- 推理继续使用 `tokenizer.apply_chat_template`（TinyLlama 兼容此格式）

**`UNIFIED_SYSTEM_PROMPT` 修正**：
- 将 `Hello.` 示例从 `{"type":"invalid"}` 改为 `{"type":"general_qa","answer":"Hello! How can I help?"}`
- 规则说明行补充：`general_qa: greetings, personal questions — never classify as invalid`
- 新增示例：
  ```
  Input: Cathey, lower the temperature.
  Output: {"type":"needs_clarification","question":"What temperature would you like?","options":["lower_ac_temperature","raise_ac_temperature"]}
  ```

### `agent.py`

**`_do_general_qa`**：
- 删除 `if pre_answer` 快捷路径
- 一律调用 `answer_qa(text, build_context(text))`，确保完整记忆（含 prefs）进入回答生成

**`_handle_new_request`**：
- context 构建时将 `self._memory.prefs` 拼入，使分类阶段也能看到 `user_name` 等偏好：
  ```python
  pref_lines = [f"- {k}: {v}" for k, v in self._memory.prefs.items()]
  context = "\n".join(ep_lines + pref_lines)
  ```

### `lora_training.ipynb`

**推理格式修正**：
- `run_inference` 改用 `tokenizer.apply_chat_template` 构建 prompt，与训练格式统一（当前用纯文本 `Input:/Output:`，与训练不一致）

**Section 6.5 数据修正**：
- `"Hello." → invalid` 改为 `general_qa`（answer: "Hello! How can I help you?"）
- `"Never mind." → invalid` 改为 `general_qa`（answer: "No problem, let me know if you need anything."）
- 新增训练样本：`"Cathey, lower the temperature."` → `needs_clarification`
- 新增训练样本：`"Cathey, what's my name?"` → `general_qa`（供 answer_qa 在有 context 时回答）

**训练后保存**：
- 确保有 cell 执行 `trainer.model.save_pretrained(FINAL_ADAPTER_DIR)` + `tokenizer.save_pretrained(FINAL_ADAPTER_DIR)`
- 新增推理验证 cell（覆盖上述 bug case）

---

## 数据流变化

### general_qa 记忆召回（修复后）

```
用户: "Cathey, what's my name?"
→ _handle_new_request
→ context = episodic_lines + pref_lines   # 含 user_name: Alex
→ parse_unified(text, context)            # 分类为 general_qa
→ _do_general_qa
→ build_context(text)                     # 完整记忆（episodic + prefs + working）
→ answer_qa(text, full_context)           # LLM 看到 user_name，正确回答
→ TTS 播报
→ save_episode()
```

---

## 不改动的部分

| 模块 | 原因 |
|------|------|
| `rule_based.py` | 快速路径逻辑正确，不受模型切换影响 |
| `schema.py` | 设备 schema 与 LLM 无关 |
| `memory.py` | 记忆架构不变，只修复调用方 |
| `audio.py` | STT/TTS 不涉及 |
| `llama_cpp` 路径 | Pi GGUF 部署暂缓，待 adapter merge 转换后处理 |

---

## 验收标准

在 `dev_debug.ipynb` 文本模式下：

| 测试输入 | 期望输出 |
|---------|---------|
| `"Hello."` | `general_qa` |
| `"Cathey, my name is Alex."` 后接 `"Cathey, what's my name?"` | `general_qa`，回答含 "Alex" |
| `"Cathey, lower the temperature."` | `needs_clarification`（通过 LLM 或 rescue path） |
| `"Cathey, I feel cold."` | `needs_clarification` |
| `"Cathey, turn on the light."` | `direct_command`（rule-based，0ms LLM） |
