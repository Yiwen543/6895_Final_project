"""
Training data for LoRA fine-tuning.

RAW_TRAIN_DATA — list of (user_input, expected_json_output) tuples.
build_dataset() — formats into a HuggingFace Dataset ready for SFTTrainer.
"""

from datasets import Dataset
from llm_parser import UNIFIED_SYSTEM_PROMPT

RAW_TRAIN_DATA = [
    # ── direct_command ───────────────────────────────────────────────────────
    ("Nova, turn on the light.",
     '{"type":"direct_command","device":"light","action":"turn_on","value":null}'),
    ("Nova, turn off the light.",
     '{"type":"direct_command","device":"light","action":"turn_off","value":null}'),
    ("Hello Nova, switch on the light please.",
     '{"type":"direct_command","device":"light","action":"turn_on","value":null}'),
    ("Nova, turn the light off.",
     '{"type":"direct_command","device":"light","action":"turn_off","value":null}'),
    ("Nova, set the brightness to 50.",
     '{"type":"direct_command","device":"light","action":"set_brightness","value":50}'),
    ("Nova, dim the light to 30 percent.",
     '{"type":"direct_command","device":"light","action":"set_brightness","value":30}'),
    ("Nova, set brightness 80.",
     '{"type":"direct_command","device":"light","action":"set_brightness","value":80}'),
    ("Nova, start RGB mode.",
     '{"type":"direct_command","device":"light","action":"rgb_cycle","value":null}'),
    ("Nova, turn on the AC.",
     '{"type":"direct_command","device":"ac","action":"turn_on","value":null}'),
    ("Nova, turn off the AC.",
     '{"type":"direct_command","device":"ac","action":"turn_off","value":null}'),
    ("Nova, set the AC to 24 degrees.",
     '{"type":"direct_command","device":"ac","action":"set_temperature","value":24}'),
    ("Nova, set temperature to 22.",
     '{"type":"direct_command","device":"ac","action":"set_temperature","value":22}'),
    ("Nova, set the air conditioner to 26.",
     '{"type":"direct_command","device":"ac","action":"set_temperature","value":26}'),
    ("Nova, AC temperature 20.",
     '{"type":"direct_command","device":"ac","action":"set_temperature","value":20}'),
    ("Nova, open the curtain.",
     '{"type":"direct_command","device":"curtain","action":"open","value":null}'),
    ("Nova, close the curtain.",
     '{"type":"direct_command","device":"curtain","action":"close","value":null}'),
    ("Nova, set curtain to 50 percent.",
     '{"type":"direct_command","device":"curtain","action":"set_position","value":50}'),
    ("Nova, open the window.",
     '{"type":"direct_command","device":"window","action":"open","value":null}'),
    ("Nova, close the window.",
     '{"type":"direct_command","device":"window","action":"close","value":null}'),
    ("Nova, please close the window.",
     '{"type":"direct_command","device":"window","action":"close","value":null}'),
    ("Nova, open window halfway.",
     '{"type":"direct_command","device":"window","action":"set_position","value":50}'),

    # ── needs_clarification ──────────────────────────────────────────────────
    ("Nova, I feel cold.",
     '{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"]}'),
    ("Nova, I feel a little bit cold.",
     '{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"]}'),
    ("Nova, it is freezing in here.",
     '{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"]}'),
    ("Nova, I am a bit chilly.",
     '{"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"]}'),
    ("Nova, I feel hot.",
     '{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"]}'),
    ("Nova, it is too warm in here.",
     '{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"]}'),
    ("Nova, it is stuffy.",
     '{"type":"needs_clarification","question":"Would you like me to open the window or lower the AC temperature?","options":["open_window","lower_ac_temperature"]}'),
    ("Nova, it is a bit dark.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"]}'),
    ("Nova, it is dark in here.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"]}'),
    ("Nova, this room is too dark.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"]}'),
    ("Nova, I can barely see anything.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"]}'),
    ("Nova, it is too bright.",
     '{"type":"needs_clarification","question":"Would you like me to dim the light or close the curtain?","options":["dim_light","close_curtain"]}'),
    ("Nova, the light is too strong.",
     '{"type":"needs_clarification","question":"Would you like me to dim the light or close the curtain?","options":["dim_light","close_curtain"]}'),
    # hard cases: colloquial / emotional
    ("Nova, fuck this light.",
     '{"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"]}'),
    ("Nova, this light is annoying.",
     '{"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"]}'),
    ("Nova, I hate this light.",
     '{"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"]}'),
    ("Nova, make this room lively.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"]}'),
    ("Nova, it is boring in here.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"]}'),
    ("Nova, spice up this room.",
     '{"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"]}'),
    ("Nova, I am not comfortable.",
     '{"type":"needs_clarification","question":"Could you describe what feels uncomfortable — temperature, lighting, or something else?","options":["adjust_temperature","adjust_lighting"]}'),

    # ── general_qa ───────────────────────────────────────────────────────────
    ("Nova, how do I eat an apple?",
     '{"type":"general_qa","answer":"Wash it first, then eat it."}'),
    ("Nova, how can I eat an apple?",
     '{"type":"general_qa","answer":"Rinse it under water and eat it directly, or peel it first if you prefer."}'),
    # hard case: mentions dish/fridge → general knowledge, NOT device control
    ("Nova, can I still eat this dish after a night in the fridge?",
     '{"type":"general_qa","answer":"Yes, most cooked food is safe for up to 3-4 days in the fridge."}'),
    ("Nova, is it safe to reheat leftover rice?",
     '{"type":"general_qa","answer":"Yes, reheat it thoroughly until steaming. Avoid reheating more than once."}'),
    ("Nova, how long does milk last in the fridge?",
     '{"type":"general_qa","answer":"Typically 5-7 days after opening if kept below 4 degrees Celsius."}'),
    ("Nova, what time is it?",
     '{"type":"general_qa","answer":"I do not have access to real-time data. Please check your phone."}'),
    ("Nova, what is the capital of France?",
     '{"type":"general_qa","answer":"The capital of France is Paris."}'),
    ("Nova, tell me a joke.",
     '{"type":"general_qa","answer":"Why do programmers prefer dark mode? Because light attracts bugs!"}'),
    ("Nova, how do I make coffee?",
     '{"type":"general_qa","answer":"Add ground coffee to a filter, pour hot water over it, and let it drip."}'),
    ("Nova, is it going to rain today?",
     '{"type":"general_qa","answer":"I do not have weather access. Please check a weather app."}'),
    ("Nova, how do I remove a stain from clothes?",
     '{"type":"general_qa","answer":"Rinse with cold water immediately, then apply detergent and wash normally."}'),
    ("Nova, how many calories are in an egg?",
     '{"type":"general_qa","answer":"A large egg contains about 70-80 calories."}'),
    ("Nova, can I drink tap water?",
     '{"type":"general_qa","answer":"It depends on your location. Check your local water quality report."}'),

    # ── invalid (no Nova prefix) ─────────────────────────────────────────────
    ("Hello.",                       '{"type":"invalid"}'),
    ("Turn on the light.",           '{"type":"invalid"}'),
    ("How are you today?",           '{"type":"invalid"}'),
    ("It is cold.",                  '{"type":"invalid"}'),
    ("What is going on?",            '{"type":"invalid"}'),
    ("Open the window.",             '{"type":"invalid"}'),
    ("Hey, can you help me?",        '{"type":"invalid"}'),
    ("I feel hot.",                  '{"type":"invalid"}'),
    ("Thanks.",                      '{"type":"invalid"}'),
    ("Bob, turn on the light.",      '{"type":"invalid"}'),
    ("Siri, turn on the light.",     '{"type":"invalid"}'),
    ("No, I feel cold.",             '{"type":"invalid"}'),
]


def build_dataset(tokenizer) -> Dataset:
    """
    Format RAW_TRAIN_DATA into a HuggingFace Dataset using the model's chat template.
    SFTTrainer computes loss only on the assistant turn (response masking is automatic).
    """
    def _fmt(inp: str, out: str) -> str:
        msgs = [
            {"role": "system",    "content": UNIFIED_SYSTEM_PROMPT},
            {"role": "user",      "content": f'Text: "{inp}"\nReturn JSON only.'},
            {"role": "assistant", "content": out},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    return Dataset.from_dict({"text": [_fmt(i, o) for i, o in RAW_TRAIN_DATA]})
