"""Quick LLM smoke test — covers all four output types + latency report.

Usage on Pi:
    sudo systemctl stop nova        # free up ~3GB before loading LLM
    cd ~/nova && python3 test_llm.py
    sudo systemctl start nova       # restore service when done
"""

import time
from llm_parser import LLMParser


TESTS = [
    # (label, text, expected_type)
    ("direct_command — light on",      "Nova, turn on the light",            "direct_command"),
    ("direct_command — AC temp",       "Nova, set the AC to 24 degrees",     "direct_command"),
    ("direct_command — close curtain", "Nova, close the curtain",            "direct_command"),
    ("needs_clarification — cold",     "Nova, I feel cold",                  "needs_clarification"),
    ("needs_clarification — dark",     "Nova, it's a bit dark in here",      "needs_clarification"),
    ("general_qa — food",              "Nova, how do I make scrambled eggs", "general_qa"),
    ("general_qa — cleaning",          "Nova, how do I clean a sofa",        "general_qa"),
    ("invalid — greeting",             "Hello",                              "invalid"),
]


def main() -> None:
    print("=" * 70)
    print("Loading LLM (this takes 10–20s on first load) ...")
    t0 = time.perf_counter()
    llm = LLMParser()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"LLM loaded in {load_ms:.0f} ms")
    print("=" * 70)

    passed = 0
    total_ms = 0.0

    for label, text, expected in TESTS:
        result, raw, ms = llm.parse_unified(text)
        actual = result.get("type", "?")
        total_ms += ms
        ok = actual == expected
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] {label}")
        print(f"  input    : {text!r}")
        print(f"  expected : {expected}")
        print(f"  got      : {actual}")
        print(f"  result   : {result}")
        print(f"  latency  : {ms:.0f} ms")

    print("\n" + "=" * 70)
    print(f"Result: {passed}/{len(TESTS)} passed")
    print(f"Average latency: {total_ms / len(TESTS):.0f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
