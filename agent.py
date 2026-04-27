"""
NovaAgent: stateful agent that takes STT text and drives the full pipeline.

Dependencies injected at construction time:
  llm    — LLMParser  (classification + QA generation)
  memory — MemoryManager (4-layer memory + RAG context builder)
  speak  — callable(text: str)  (TTS or print)

Four intent types handled:
  direct_command         — execute device action immediately
  needs_clarification    — ask user to pick an option; auto-resolve via procedural memory
  general_qa             — answer with RAG-augmented context
  invalid                — no response (no "Nova" prefix or empty)
"""

import threading
from gpio_executor import GPIOExecutor
from rule_based import try_rule_based
from typing import Any, Callable, Dict, Optional

from config import ASSISTANT_NAME_VARIANTS
from schema import (
    build_clarification_reply,
    execute_command,
    option_to_display,
    validate_command,
)


def contains_assistant_name(text: str) -> bool:
    t = text.strip().lower()
    return any(name in t for name in ASSISTANT_NAME_VARIANTS)


class NovaAgent:

    def __init__(self, llm, memory, speak: Callable[[str], None],
                 gpio: GPIOExecutor = None):
        self._llm    = llm
        self._memory = memory
        self._speak  = speak
        self._gpio   = gpio
        self._state  = self._blank_state()

    @staticmethod
    def _blank_state() -> Dict:
        return {
            "pending_clarification": False,
            "original_text":         None,
            "question":              None,
            "options":               None,
        }

    def reset_dialogue(self):
        self._state = self._blank_state()

    @staticmethod
    def _rule_reply(semantic: dict) -> str:
        device = semantic.get("device", "")
        action = semantic.get("action", "")
        value  = semantic.get("value")
        if action == "turn_on":
            return f"Sure, turning on the {device}."
        if action == "turn_off":
            return f"Sure, turning off the {device}."
        if action == "set_brightness":
            return f"Sure, setting brightness to {value} percent."
        if action == "rgb_cycle":
            return "Sure, starting RGB cycle."
        if action == "open":
            return f"Sure, opening the {device}."
        if action == "close":
            return f"Sure, closing the {device}."
        if action == "set_position":
            return f"Sure, setting {device} to {value} percent."
        if action == "set_temperature":
            return f"Sure, setting AC to {value} degrees."
        return "Done."

    # ── Main entry point ──────────────────────────────────────────────────────

    def handle(self, text: str, verbose: bool = True) -> Dict[str, Any]:
        text = text.strip()
        if verbose:
            print("STT text:", text)

        if self._state["pending_clarification"]:
            return self._handle_followup(text, verbose)
        return self._handle_new_request(text, verbose)

    # ── Follow-up reply (clarification pending) ───────────────────────────────

    def _handle_followup(self, text: str, verbose: bool) -> Dict[str, Any]:
        if not text:
            return self._result(True, {"type": "invalid"}, False, "empty_clarification_reply")

        self._memory.push_working("user", text)
        semantic, _, ms = self._llm.resolve_followup(
            self._state["original_text"],
            self._state["question"],
            self._state["options"],
            text,
            verbose=verbose,
        )

        if semantic["type"] == "direct_command":
            cmd = {k: semantic[k] for k in ("device", "action", "value")}
            ok, reason = validate_command(cmd)
            if ok:
                reply = semantic.get("reply") or "Done."
                self._speak(reply)
                self._memory.record_skill(self._state["original_text"], cmd)
                self._update_pref(cmd)
                self._memory.save_episode(self._state["original_text"], "needs_clarification", reply)
                self._memory.push_working("nova", reply)
                self.reset_dialogue()
                return self._result(True, semantic, True, reason,
                                    execute_command(cmd), reply, round(ms, 3))

            reply = "Sorry, I could not resolve that action safely."
            self._speak(reply)
            self._memory.push_working("nova", reply)
            self.reset_dialogue()
            return self._result(True, semantic, False, reason, "SKIPPED", reply, round(ms, 3))

        reply = "Sorry, I didn't catch your choice. Please answer again."
        self._speak(reply)
        self._memory.push_working("nova", reply)
        return self._result(True, {"type": "invalid"}, False,
                            "clarification_not_resolved", "SKIPPED", reply, round(ms, 3))

    # ── New user request ──────────────────────────────────────────────────────

    def _handle_new_request(self, text: str, verbose: bool) -> Dict[str, Any]:
        if not text:
            return self._result(False, {"type": "invalid"}, False, "empty_text")

        if not contains_assistant_name(text):
            if verbose:
                print("Assistant name not detected. Skipping.")
            return self._result(False, {"type": "invalid"}, False, "assistant_name_not_detected")

        self._memory.push_working("user", text)

        # Rule-based fast path: skip LLM for unambiguous direct commands
        fast = try_rule_based(text)
        if fast is not None:
            fast["reply"] = self._rule_reply(fast)
            return self._do_direct_command(fast, text, 0.0)

        semantic, _, ms = self._llm.parse_unified(text, verbose=verbose)

        if semantic["type"] == "direct_command":
            return self._do_direct_command(semantic, text, ms)

        if semantic["type"] == "needs_clarification":
            return self._do_clarification(semantic, text, ms, verbose)

        if semantic["type"] == "general_qa":
            return self._do_general_qa(text, ms, verbose)

        reply = "Sorry, I didn't understand that."
        self._speak(reply)
        self._memory.push_working("nova", reply)
        return self._result(True, {"type": "invalid"}, False,
                            "invalid_semantic_result", "SKIPPED", reply, round(ms, 3))

    # ── Intent handlers ───────────────────────────────────────────────────────

    def _do_direct_command(self, semantic, text, ms) -> Dict[str, Any]:
        cmd = {k: semantic[k] for k in ("device", "action", "value")}
        ok, reason = validate_command(cmd)
        if ok:
            reply = semantic.get("reply") or "Done."
            hw_result = execute_command(cmd)

            # GPIO and TTS run concurrently
            gpio_thread = threading.Thread(
                target=self._gpio.execute, args=(cmd,), daemon=True
            ) if self._gpio else None
            tts_thread = threading.Thread(
                target=self._speak, args=(reply,), daemon=True
            )
            if gpio_thread:
                gpio_thread.start()
            tts_thread.start()
            if gpio_thread:
                gpio_thread.join()
            tts_thread.join()

            self._update_pref(cmd)
            self._memory.save_episode(text, "direct_command", reply)
            self._memory.push_working("nova", reply)
            return self._result(True, semantic, True, reason,
                                hw_result, reply, round(ms, 3))

        self._memory.push_working("nova", "(command invalid)")
        return self._result(True, semantic, False, reason, "SKIPPED", None, round(ms, 3))

    def _do_clarification(self, semantic, text, ms, verbose) -> Dict[str, Any]:
        # Check procedural memory for a known preference
        known = self._memory.lookup_skill(text)
        if known:
            cmd = known["action"]
            ok, reason = validate_command(cmd)
            if ok:
                action_label = f"{cmd['device']} {cmd['action']}".replace("_", " ")
                reply = f"Sure, {action_label}. (based on your past preference)"
                self._speak(reply)
                self._memory.record_skill(text, cmd)
                self._update_pref(cmd)
                self._memory.save_episode(text, "needs_clarification_auto", reply)
                self._memory.push_working("nova", reply)
                if verbose:
                    print(f"[Procedural Memory] Auto-resolved: {cmd}")
                return self._result(True, semantic, True, "procedural_memory_auto_resolved",
                                    execute_command(cmd), reply, round(ms, 3))

        # No known preference — ask for clarification
        self._state.update({
            "pending_clarification": True,
            "original_text":         text,
            "question":              semantic["question"],
            "options":               semantic["options"],
        })
        clarification = build_clarification_reply(semantic["question"], semantic["options"])
        if verbose:
            print("[Clarification options]")
            for i, opt in enumerate(semantic["options"], 1):
                print(f"  [{i}] {option_to_display(opt)}")
        self._speak(clarification)
        self._memory.push_working("nova", clarification)
        return self._result(True, semantic, True, "clarification_requested",
                            "PENDING_USER_REPLY", clarification, round(ms, 3))

    def _do_general_qa(self, text, ms, verbose) -> Dict[str, Any]:
        context = self._memory.build_context(text)
        answer, qa_ms = self._llm.answer_qa(text, context, verbose=verbose)
        self._speak(answer)
        self._memory.save_episode(text, "general_qa", answer)
        self._memory.push_working("nova", answer)
        return self._result(True, {"type": "general_qa", "answer": answer}, True,
                            "general_qa_answered", "NO_DEVICE_ACTION", answer,
                            round(ms + qa_ms, 3))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_pref(self, cmd: Dict):
        if cmd["value"] is not None:
            self._memory.update_pref(
                f"{cmd['device']}_{cmd['action']}_preference", cmd["value"]
            )

    @staticmethod
    def _result(
        prefilter: bool,
        semantic: Dict,
        valid: bool,
        reason: str,
        execution: str = "SKIPPED",
        spoken_reply: Optional[str] = None,
        llm_latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        r: Dict[str, Any] = {
            "prefilter_passed": prefilter,
            "semantic":         semantic,
            "valid":            valid,
            "reason":           reason,
            "execution":        execution,
        }
        if spoken_reply is not None:
            r["spoken_reply"] = spoken_reply
        if llm_latency_ms is not None:
            r["llm_latency_ms"] = llm_latency_ms
        return r
