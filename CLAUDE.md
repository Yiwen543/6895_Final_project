# Nova Smart Home Assistant — CLAUDE.md

EECS 6895 Final Project. Nova 是一个运行在树莓派 5 上的本地语音智能家居助手，
使用 Whisper 做 STT、TinyLlama（LoRA 微调）做意图解析、Piper 做 TTS。

---

## 项目文件

| 文件 | 说明 |
|------|------|
| `Nova_4_16.ipynb` | 主 pipeline：录音 → STT → LLM → 设备控制 |
| `lora_training.ipynb` | TinyLlama LoRA 微调训练脚本 |
| `model_comparison.py` | 三模型横向对比评测（TinyLlama / Qwen2.5 / Phi-2）|
| `OPTIMIZATION_SUGGESTIONS.md` | 已诊断的性能问题与修复方案（必读）|

---

## 模型与推理配置

- **STT**：`faster-whisper tiny.en`（int8，CPU）
- **LLM**：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`（当前 float32 CPU，目标换 GGUF Q4_K_M）
- **LoRA adapter 路径**：`./tinyllama_home_lora/final_adapter`
- **设备检测**：`cuda > mps > cpu`（树莓派上为 cpu）
- **推理框架**：HuggingFace transformers（本地开发）/ llama-cpp-python（树莓派部署目标）

---

## LLM 输出格式（意图解析）

模型只输出以下四种 JSON，不得有额外文本：

```json
{"type":"direct_command","device":"light|curtain|window|ac","action":"turn_on|turn_off|set_brightness|rgb_cycle|open|close|set_position|set_temperature","value":null_or_int}
{"type":"needs_clarification","question":"...","options":["...","..."]}
{"type":"general_qa","answer":"..."}
{"type":"invalid"}
```

**分类规则**：
- `direct_command`：用户明确说出设备 + 动作（"turn on the light"）
- `needs_clarification`：描述感受或氛围，未指定具体动作（"I feel cold"、"it's dark"）
- `general_qa`：与家居设备无关的问题（"how do I eat an apple"）
- `invalid`：无有效意图（"hello"、"never mind"）

---

## 已知问题与修复状态

> 详见 `OPTIMIZATION_SUGGESTIONS.md`，以下为快速参考。

| # | 问题 | 影响 | 状态 |
|---|------|------|------|
| 1 | LoRA adapter 训练后未被加载 | 准确率 +30–40% | ⬜ 待修复 |
| 2 | 训练与推理的 system prompt 不一致 | 延迟 −30%，准确率 +10% | ⬜ 待修复 |
| 3 | LLM 在 CPU 上跑 float32（~18s/次） | 延迟 −95%（规则）/ −80%（GGUF） | ⬜ 待修复 |
| 4 | 每次 STT 都有磁盘 I/O | 延迟 −50–100ms | ⬜ 待修复 |
| 5 | 唤醒词 "Nova" 识别变体太少 | 减少漏触发 | ⬜ 待修复 |
| 6 | LoRA 训练集缺少 general_qa / invalid 样本 | 准确率 +15–20% | ⬜ 待修复 |
| 7 | 用 llama-cpp-python 替换 HF 推理 | 延迟 −80% | ⬜ 待修复 |

**当前基准**：Exact Match 33%，平均延迟 ~18,400ms  
**修复 1–5 后预期**：准确率 ~80%+，直接指令延迟 <10ms

---

## 常用命令

```bash
# 运行模型对比评测
python model_comparison.py

# 启动主 pipeline（Jupyter）
jupyter notebook Nova_4_16.ipynb

# 启动 LoRA 训练
jupyter notebook lora_training.ipynb

# 查看对比日志
cat comparison_log.txt
```

---

## 部署目标：树莓派 5

- **系统**：Raspberry Pi OS 64-bit（ARM64）
- **目标模型格式**：GGUF Q4_K_M（通过 llama-cpp-python 推理）
- **推理线程**：`n_threads=4`，`n_ctx=512`
- **同步命令**：`rsync -avz ./ pi@<PI_IP>:~/nova/`
- **服务管理**：systemd `nova.service`

---

## 开发注意事项

- 修改 system prompt 时，必须同步更新 `lora_training.ipynb` 和 `Nova_4_16.ipynb`，保持格式一致
- LoRA adapter 加载需要在 base model 加载后立即执行（用 `peft.PeftModel.from_pretrained`）
- 树莓派上禁用 float32 推理，只用 GGUF Q4_K_M
- 唤醒词变体列表：`["nova", "nava", "no va", "noba", "noa", "nove", "novia"]`
- 音频采样率统一 `16000 Hz`
- STT 转录后直接用 numpy buffer，不写磁盘临时文件
