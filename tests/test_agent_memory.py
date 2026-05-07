from llm_parser import UNIFIED_SYSTEM_PROMPT


def test_hello_is_not_invalid_example():
    """The system prompt must not teach the model that 'Hello' is invalid."""
    assert '"type":"invalid"' not in UNIFIED_SYSTEM_PROMPT.split("Hello")[1][:60]


def test_prompt_teaches_hello_as_general_qa():
    assert "general_qa" in UNIFIED_SYSTEM_PROMPT
    assert "greetings" in UNIFIED_SYSTEM_PROMPT.lower()


def test_prompt_has_lower_temperature_example():
    assert "lower the temperature" in UNIFIED_SYSTEM_PROMPT.lower()
