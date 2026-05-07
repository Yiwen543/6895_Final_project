from llm_parser import UNIFIED_SYSTEM_PROMPT


def test_hello_is_not_invalid_example():
    """The system prompt must not teach the model that 'Hello' is invalid."""
    assert '"type":"invalid"' not in UNIFIED_SYSTEM_PROMPT.split("Hello")[1][:60]


def test_prompt_teaches_hello_as_general_qa():
    assert "general_qa" in UNIFIED_SYSTEM_PROMPT
    assert "greetings" in UNIFIED_SYSTEM_PROMPT.lower()


def test_prompt_has_lower_temperature_example():
    assert "lower the temperature" in UNIFIED_SYSTEM_PROMPT.lower()


from unittest.mock import MagicMock
from agent import CatheyAgent


def _make_agent(parse_result, qa_result=""):
    llm = MagicMock()
    llm.parse_unified.return_value = (parse_result, "", 100.0)
    llm.answer_qa.return_value = (qa_result, 50.0)

    memory = MagicMock()
    memory.episodes.count.return_value = 0
    memory.prefs = {"user_name": "Alex"}
    memory.build_context.return_value = "## User preferences\n- user_name: Alex"
    memory.skills = []

    speak = MagicMock()
    return CatheyAgent(llm=llm, memory=memory, speak=speak, gpio=None), llm, memory, speak


def test_prefs_included_in_parse_unified_context():
    """parse_unified must receive context that contains current user prefs."""
    agent, llm, memory, speak = _make_agent(
        {"type": "general_qa", "answer": "Your name is Alex."}, "Your name is Alex."
    )
    agent.handle("Cathey, what's my name?", verbose=False)

    _, kwargs = llm.parse_unified.call_args
    context = kwargs.get("context", "")
    assert "user_name" in context
    assert "Alex" in context


def test_general_qa_always_calls_answer_qa():
    """answer_qa must always be called for general_qa, never using the classification answer directly."""
    agent, llm, memory, speak = _make_agent(
        {"type": "general_qa", "answer": "classification-side answer"},
        "memory-aware answer"
    )
    agent.handle("Cathey, what is your name?", verbose=False)

    llm.answer_qa.assert_called_once()
    speak.assert_called_once_with("memory-aware answer")
