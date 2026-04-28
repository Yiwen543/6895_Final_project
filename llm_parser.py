"""
LLM module: loads a configurable instruction-tuned model (default: Qwen2.5-3B-Instruct)
and exposes three inference methods:
  - parse_unified()    classify user text → direct_command / needs_clarification / general_qa / invalid
  - resolve_followup() resolve user reply to a clarification question → direct_command / invalid
  - answer_qa()        generate a plain-text answer using RAG context

On Mac (MPS/CUDA): uses HuggingFace transformers with float16.
On Pi 5 (CPU): uses llama-cpp-python with a GGUF Q4_K_M quantized model.
Backend is selected via LLM_BACKEND in config.py.
"""

import re
import json
import time
import torch
from typing import Any, Dict, List, Optional, Tuple

from config import LLM_BACKEND, LLM_MODEL_NAME, LLM_GGUF_PATH, LLM_DEVICE, LLM_DTYPE, LLM_MAX_NEW_TOKENS

if LLM_BACKEND == "transformers":
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )
else:
    from llama_cpp import Llama


# ── System prompts ────────────────────────────────────────────────────────────

UNIFIED_SYSTEM_PROMPT = """
Return exactly one JSON object and nothing else.

Allowed outputs:

{"type":"direct_command","device":"light|curtain|window|ac","action":"turn_on|turn_off|set_brightness|rgb_cycle|open|close|set_position|set_temperature","value":null_or_int,"reply":"brief natural-language confirmation"}

{"type":"needs_clarification","question":"...","options":["...","..."],"reply":"the question restated naturally for speech"}

{"type":"general_qa","answer":"..."}

{"type":"invalid"}

CRITICAL classification rules:

direct_command: ONLY when the user explicitly names BOTH a device (light/curtain/window/ac) AND a specific action (turn on/off, open/close, set to X). No ambiguity allowed.
  → reply: a short, friendly confirmation of the action taken. e.g. "Sure, turning on the light!"

needs_clarification: When the user describes how they FEEL about the home environment (cold, hot, dark, bright, stuffy, boring) OR expresses frustration, annoyance, or a vague atmosphere preference about a home device — WITHOUT specifying a concrete action.
  → reply: the clarification question in natural speech form.

general_qa: When the user asks about ANY topic NOT about controlling home devices. Includes food, cooking, eating, food safety, health, science, weather, time, general knowledge.

invalid: When there is no meaningful request.

DO NOT use direct_command for vague feelings or environmental descriptions — always needs_clarification.
DO NOT use needs_clarification for food, cooking, eating, or general knowledge questions — always general_qa.
DO NOT use general_qa for complaints or feelings about the home environment — always needs_clarification.

Examples:

Input: Nova, turn on the light.
Output: {"type":"direct_command","device":"light","action":"turn_on","value":null,"reply":"Sure, turning on the light!"}

Input: Nova, turn off the light.
Output: {"type":"direct_command","device":"light","action":"turn_off","value":null,"reply":"Got it, light is off."}

Input: Nova, set the AC to 24 degrees.
Output: {"type":"direct_command","device":"ac","action":"set_temperature","value":24,"reply":"Setting the AC to 24 degrees."}

Input: Nova, open the curtain.
Output: {"type":"direct_command","device":"curtain","action":"open","value":null,"reply":"Opening the curtain for you."}

Input: Nova, close the window.
Output: {"type":"direct_command","device":"window","action":"close","value":null,"reply":"Closing the window now."}

Input: Nova, I feel cold.
Output: {"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}

Input: Nova, I feel a little bit cold.
Output: {"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"],"reply":"Would you like me to close the window or raise the AC temperature?"}

Input: Nova, it's a bit dark.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}

Input: Nova, this room is too dark.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"],"reply":"Would you like me to turn on the light or open the curtain?"}

Input: Nova, I feel hot.
Output: {"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"],"reply":"Would you like me to open the window or lower the AC temperature?"}

Input: Nova, fuck this light.
Output: {"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"],"reply":"Would you like me to turn off the light or dim it?"}

Input: Nova, this light is annoying.
Output: {"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"],"reply":"Would you like me to turn off the light or dim it?"}

Input: Nova, make this room lively.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"],"reply":"Would you like me to turn on the RGB cycle or open the curtain?"}

Input: Nova, it's boring in here.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"],"reply":"Would you like me to turn on the RGB cycle or open the curtain?"}

Input: Nova, how do I eat an apple?
Output: {"type":"general_qa","answer":"Wash it first, then eat it."}

Input: Nova, can I still eat this dish after a night in the fridge?
Output: {"type":"general_qa","answer":"Yes, most cooked food is safe for up to 3-4 days in the fridge."}

Input: Nova, is it safe to reheat leftover rice?
Output: {"type":"general_qa","answer":"Yes, reheat it thoroughly until steaming hot. Avoid reheating more than once."}

Input: Nova, what time is it?
Output: {"type":"general_qa","answer":"I don't have access to real-time data, but you can check your phone."}

Input: Hello.
Output: {"type":"invalid"}
""".strip()

FOLLOWUP_RESOLUTION_SYSTEM_PROMPT = """
You are resolving the user's reply to a previous clarification question.

Return exactly one JSON object and nothing else.

Allowed output types:

1. direct_command
{"type":"direct_command","device":"light|curtain|window|ac","action":"turn_on|turn_off|set_brightness|rgb_cycle|open|close|set_position|set_temperature","value":null_or_int,"reply":"brief natural-language confirmation"}

2. invalid
{"type":"invalid"}

Rules:
- Use the previous user request, the clarification question, the available options, and the current reply.
- If the reply clearly selects one option, return direct_command with a friendly reply field.
- If the reply is unclear, return invalid.
- No explanation. No markdown. No extra text.
""".strip()

QA_SYSTEM_PROMPT = """
You are Nova, a helpful smart home assistant.
Answer the user's question concisely in 1-2 sentences.
Use the provided context only if it is clearly relevant.
Do NOT mention devices (light/curtain/window/AC) unless asked.
Reply in plain text only — no JSON, no markdown.
""".strip()


# ── Stopping criterion (transformers backend only) ───────────────────────────

if LLM_BACKEND == "transformers":
    class _JsonStop(StoppingCriteria):
        def __init__(self, tokenizer, prompt_length: int):
            self._tokenizer = tokenizer
            self._prompt_len = prompt_length

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            gen = input_ids[0][self._prompt_len:]
            text = self._tokenizer.decode(gen, skip_special_tokens=True)
            return LLMParser._has_complete_json(text)


# ── Main class ────────────────────────────────────────────────────────────────

class LLMParser:
    """Loads the LLM once; exposes parse_unified / resolve_followup / answer_qa."""

    def __init__(self, model_name: str = LLM_MODEL_NAME, dtype=LLM_DTYPE):
        self._backend = LLM_BACKEND

        if self._backend == "llama_cpp":
            print(f"Loading LLM via llama.cpp ({LLM_GGUF_PATH}) ...")
            self._llama = Llama(
                model_path=LLM_GGUF_PATH,
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            self.tokenizer = None
            self.model = None
        else:
            print(f"Loading LLM ({model_name}) on {LLM_DEVICE} [{dtype}] ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if LLM_DEVICE == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype
                ).to("mps")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype, device_map="auto"
                )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            self._llama = None
        print("LLM ready.")

    # ── Static JSON helpers ───────────────────────────────────────────────────

    @staticmethod
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
                    return text[start : i + 1]
        return None

    @staticmethod
    def _has_complete_json(text: str) -> bool:
        start = text.find("{")
        if start == -1:
            return False
        depth = 0
        for ch in text[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False

    @staticmethod
    def _normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            return {"type": "invalid"}
        t = str(obj.get("type", "invalid")).strip().lower()

        if t == "direct_command":
            device = obj.get("device")
            action = obj.get("action")
            value  = obj.get("value")
            if not isinstance(device, str) or not isinstance(action, str):
                return {"type": "invalid"}
            device = device.strip().lower()
            action = action.strip().lower()
            if "|" in device or "|" in action:
                return {"type": "invalid"}
            if value in ("null", "None", "none", "NULL"):
                value = None
            if isinstance(value, str):
                v = value.strip().replace("%", "")
                if re.fullmatch(r"\d{1,3}", v):
                    value = int(v)
            reply = str(obj.get("reply", "")).strip()
            return {"type": "direct_command", "device": device, "action": action,
                    "value": value, "reply": reply}

        if t == "needs_clarification":
            question = obj.get("question", "")
            options  = obj.get("options", [])
            if not isinstance(question, str) or not isinstance(options, list):
                return {"type": "invalid"}
            options = [str(x).strip() for x in options if str(x).strip()]
            if not options:
                return {"type": "invalid"}
            reply = str(obj.get("reply", question)).strip()
            return {"type": "needs_clarification", "question": question.strip(),
                    "options": options, "reply": reply}

        if t == "general_qa":
            answer = obj.get("answer", "")
            if not isinstance(answer, str):
                return {"type": "invalid"}
            return {"type": "general_qa", "answer": answer.strip()}

        return {"type": "invalid"}

    # ── Core generation ───────────────────────────────────────────────────────

    def _generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], str, float]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        if self._backend == "llama_cpp":
            t0 = time.perf_counter()
            resp = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
                repeat_penalty=1.1,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = resp["choices"][0]["message"]["content"].strip()
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            plen = inputs["input_ids"].shape[1]

            stop = StoppingCriteriaList([_JsonStop(self.tokenizer, plen)])
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=True,
                    stopping_criteria=stop,
                )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = self.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()

        if verbose:
            print("Raw output:", raw)
            print(f"Latency: {latency_ms:.1f} ms")

        json_str = self._extract_first_json(raw)
        if json_str is None:
            return {"type": "invalid"}, raw, latency_ms
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return {"type": "invalid"}, raw, latency_ms
        return self._normalize(parsed), raw, latency_ms

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_unified(
        self, text: str, verbose: bool = False
    ) -> Tuple[Dict[str, Any], str, float]:
        return self._generate_json(
            UNIFIED_SYSTEM_PROMPT,
            f'Text: "{text}"\nReturn JSON only.',
            verbose=verbose,
        )

    def resolve_followup(
        self,
        original_text: str,
        question: str,
        options: List[str],
        user_reply: str,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], str, float]:
        opts = "\n".join(f"- {o}" for o in options)
        user_prompt = (
            f'Original request: "{original_text}"\n'
            f'Clarification question: "{question}"\n'
            f'Available options:\n{opts}\n'
            f'User reply: "{user_reply}"\n\nReturn JSON only.'
        )
        return self._generate_json(FOLLOWUP_RESOLUTION_SYSTEM_PROMPT, user_prompt, verbose=verbose)

    def answer_qa(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 80,
        verbose: bool = False,
    ) -> Tuple[str, float]:
        user_content = f"{context}\n\nUser: {question}" if context.strip() else f"User: {question}"
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        if self._backend == "llama_cpp":
            t0 = time.perf_counter()
            resp = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = resp["choices"][0]["message"]["content"].strip()
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            plen = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = self.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()

        if verbose:
            print("[QA answer]", answer)
        return answer, latency_ms

    # ── LoRA utilities ────────────────────────────────────────────────────────

    def load_lora_adapter(self, adapter_dir: str):
        """Load a saved LoRA adapter on top of the base model."""
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_dir} ...")
        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        print("Adapter loaded.")

    def merge_lora(self):
        """Merge LoRA A/B matrices into base weights and remove adapter overhead."""
        print("Merging LoRA weights ...")
        self.model = self.model.merge_and_unload()
        self.model.eval()
        print("Merged.")

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}/")
