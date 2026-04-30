# Nova Smart Home Assistant — CLAUDE.md

EECS 6895 Final Project. Nova 是一个运行在树莓派 5 上的本地语音智能家居助手，
完全本地推理（无云端），使用 faster-whisper 做 STT、Qwen2.5-3B-Instruct（GGUF Q3_K_M）做意图解析、Piper 做 TTS。

---

## 项目文件

| 文件 | 说明 |
|------|------|
| `nova.py` | 入口，将所有组件串联 |
| `agent.py` | 有状态意图处理器（direct_command / needs_clarification / general_qa / invalid） |
| `audio.py` | STT（faster-whisper）、TTS（Piper）、VAD 录音循环（AudioListener） |
| `llm_parser.py` | LLM 推理：parse_unified / resolve_followup / answer_qa；system prompt 在此维护 |
| `config.py` | 全部配置常量 |
| `schema.py` | 设备 schema、校验、执行表 |
| `memory.py` | 四层记忆：working / episodic / procedural / preference |
| `rule_based.py` | 正则快速路径，处理无歧义直接指令（<5ms） |
| `gpio_executor.py` | GPIO 硬件执行 |
| `benchmark_quantization.py` | 独立 GGUF 量化基准测试脚本 |
| `nova.service` | systemd 服务文件 |
| `deploy.sh` | 部署脚本（rsync + 模型下载） |

> 旧版 `Nova_4_16.ipynb`、`lora_training.ipynb`、`model_comparison.py` 为遗留文件，不再是主 pipeline。

---

## 模型与推理配置

- **LLM**：Qwen2.5-3B-Instruct，GGUF Q3_K_M（`models/qwen2.5-3b-instruct-q3_k_m.gguf`）
- **推理后端**：llama-cpp-python（Pi）/ HuggingFace transformers（Mac/GPU 开发）
- **推理参数**：`n_ctx=1024`，`n_threads=4`，`temperature=0`，`LLM_MAX_NEW_TOKENS=150`
- **STT**：faster-whisper tiny.en，int8，CPU；`initial_prompt` 包含家居词表
- **TTS**：Piper，en_US-lessac-medium，170 WPM

---

## LLM 输出格式（意图解析）

模型只输出以下四种 JSON，不得有额外文本：

```json
{"type":"direct_command","device":"light|curtain|window|ac","action":"turn_on|turn_off|set_brightness|rgb_cycle|open|close|set_position|set_temperature","value":null_or_int,"reply":"brief confirmation"}
{"type":"needs_clarification","question":"...","options":["...","..."]}
{"type":"general_qa","answer":"..."}
{"type":"invalid"}
```

**分类规则**：
- `direct_command`：用户明确说出设备 + 动作（"turn on the light"）
- `needs_clarification`：描述感受/氛围（"cold"、"hot"、"dark"、"stuffy"），未指定具体设备或动作——**禁止从感受直接推断设备**
- `general_qa`：与家居设备无关的问题，包括身份问题（"what's your name?"）、问候、食物、科学、时间等
- `invalid`：**仅限**无法理解的声音或空白输入（"hello"、"never mind" 应归为 general_qa，不是 invalid）

---

## 音频 Pipeline

- **VAD**：基于能量，`ENERGY_THRESHOLD=0.05`，`SAMPLE_RATE=16000`
- **参数**：`SILENCE_SECONDS=0.5`，`MIN_SPEECH_SECONDS=0.3`，`MAX_UTTERANCE_SECONDS=8.0`
- 每次语音输入单独开关 InputStream：TTS 播放前关闭麦克风（避免回声），播放完毕后重新打开
- STT 直接从 numpy buffer 转录，不写磁盘临时文件

---

## 助手身份

- **名称**：nova
- **唤醒词变体**：`["nova", "nava", "no va", "noba", "noa", "nove", "novia", "noda", "nota", "nora", "know-a", "nana"]`

---

## 当前性能

- 规则快速路径处理约 70% 的直接指令，延迟 <5ms
- LLM 处理歧义/QA 输入，延迟 ~5–40s（取决于 prompt 长度）
- 分类准确率：~85%（3B-Q3_K_M，20 条测试集）
- 详细基准见 `benchmark_results.md` 和 `benchmark_results.csv`

---

## 部署

```bash
# 同步到 Pi（执行前必须与用户确认）
rsync -avz ./ tl3461@192.168.100.1:~/nova/

# 重启服务
sudo systemctl restart nova

# 实时日志
sudo journalctl -u nova -f
```

> **重要**：同步到 Pi 前必须先向用户确认，不得自动执行。

---

## 开发注意事项

- **System prompt 只在 `llm_parser.py`（`UNIFIED_SYSTEM_PROMPT`）中维护**，无需同步 notebook
- `n_ctx=1024`（历经 512 → 768 → 1024，以容纳完整 system prompt + 用户输入）
- `needs_clarification` 输出不含 `reply` 字段（避免 max_new_tokens 不足时 JSON 截断）
- procedural memory 自动执行已从 `_do_clarification` 移除（之前会不询问用户直接依据历史偏好执行）
- 音频采样率统一 `16000 Hz`
