"""The locally served model: backend settings, reasoning effort, captured traces."""

import pytest

from tb_search.config import ConfigError, Settings, load_settings
from tb_search.prompts import RESISTANCE_PROMPT
from tb_search.reasoning import (
    EffortError,
    answer_and_trace,
    normalize_effort,
    reasoning_trace,
    reply_text,
    split_thinking,
)
from tb_search.resistance import check_trace, normalize_prediction, predict
from tb_search.screening import (
    apply_reasoning_effort,
    is_transient_error,
    reasoning_chat_model,
)

PROVIDER_ENV = (
    "VLLM_MODEL",
    "LLM_MODEL",
    "VLLM_BASE_URL",
    "LLM_BASE_URL",
    "BIOMNI_CUSTOM_BASE_URL",
    "VLLM_API_KEY",
    "BIOMNI_CUSTOM_API_KEY",
    "LLM_SOURCE",
    "LLM_REASONING_EFFORT",
    "VLLM_REASONING_EFFORT",
)


@pytest.fixture
def clean_env(monkeypatch):
    """A bare environment: .env on disk must not decide what these tests see."""
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tb_search.config.load_dotenv", lambda *a, **k: None)


class FakeReply:
    """What langchain hands back: content, plus whatever it did not recognise."""

    def __init__(self, content, additional_kwargs=None, response_metadata=None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}
        self.response_metadata = response_metadata or {}


class FakeLLM:
    """One canned reply per invoke, and a record of the prompts it was given."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


# --- settings -----------------------------------------------------------------


def test_the_default_backend_is_a_locally_served_openai_compatible_server(clean_env):
    settings = load_settings()
    assert settings.is_custom
    assert settings.source == "Custom"
    assert settings.base_url == "http://localhost:8000/v1"
    # A local server wants no credential but the OpenAI client insists on
    # sending one.
    assert settings.api_key == "EMPTY"


def test_the_vllm_endpoint_can_come_from_the_environment(clean_env, monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu042:8000/v1")
    monkeypatch.setenv("VLLM_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    settings = load_settings()
    assert settings.base_url == "http://gpu042:8000/v1"
    assert settings.model == "Qwen/Qwen3-8B"
    assert settings.reasoning_effort == "high"
    assert "@ http://gpu042:8000/v1" in settings.description
    assert "effort high" in settings.description


def test_ollama_is_the_other_local_backend(clean_env):
    settings = load_settings(source="ollama")
    # Folded to the way biomni spells it, and reached on its own port rather
    # than by URL, so the endpoint is left out of the console line.
    assert settings.source == "Ollama"
    assert not settings.is_custom
    assert "@" not in settings.description


def test_a_hosted_backend_is_refused(clean_env):
    """Nothing here calls out to a hosted API, so asking for one is a mistake."""
    for source in ("Groq", "OpenAI", "Anthropic"):
        with pytest.raises(ConfigError, match="not a local biomni backend"):
            load_settings(source=source)


def test_a_stale_source_in_the_environment_is_ignored(clean_env, monkeypatch, capsys):
    """This pipeline used to run on Groq; a leftover .env must not break it."""
    monkeypatch.setenv("LLM_SOURCE", "Groq")
    settings = load_settings()
    assert settings.source == "Custom"
    assert "ignored" in capsys.readouterr().out


def test_a_base_url_means_the_openai_compatible_branch(clean_env, monkeypatch):
    monkeypatch.setenv("LLM_SOURCE", "Ollama")
    settings = load_settings(base_url="http://gpu042:8000/v1")
    assert settings.source == "Custom"
    assert settings.base_url == "http://gpu042:8000/v1"


def test_an_unknown_reasoning_effort_is_refused(clean_env):
    with pytest.raises(ConfigError):
        load_settings(base_url="http://localhost:8000/v1", reasoning_effort="maximum")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("high", "high"), (" NONE ", "none"), ("", None), (None, None)],
)
def test_normalize_effort(raw, expected):
    assert normalize_effort(raw) == expected


def test_normalize_effort_rejects_what_vllm_would_not_take():
    with pytest.raises(EffortError):
        normalize_effort("very high")


def test_reasoning_effort_rides_along_in_the_request_body():
    chat = apply_reasoning_effort(_chat_openai(), "high")
    assert chat.extra_body["reasoning_effort"] == "high"
    # Still a chat model, not a bound runnable: A1 calls with_structured_output.
    assert hasattr(chat, "with_structured_output")


def test_no_effort_leaves_the_model_untouched():
    chat = _chat_openai()
    assert apply_reasoning_effort(chat, None) is chat
    assert not chat.extra_body


def test_the_chat_model_keeps_the_reasoning_langchain_would_drop():
    """ChatOpenAI tracks the official schema and discards vLLM's extra field."""
    chat = reasoning_chat_model(Settings(model="m"))
    response = {
        "id": "x",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"call": "resistant"}',
                    "reasoning": "katG S315T abolishes isoniazid activation",
                },
            }
        ],
    }
    message = chat._create_chat_result(response).generations[0].message
    assert message.content == '{"call": "resistant"}'
    assert reasoning_trace(message) == "katG S315T abolishes isoniazid activation"


def test_the_older_field_name_is_kept_too():
    chat = reasoning_chat_model(Settings(model="m"))
    response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "a",
                    "reasoning_content": "b",
                },
            }
        ]
    }
    message = chat._create_chat_result(response).generations[0].message
    assert reasoning_trace(message) == "b"


def _chat_openai():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="openai/gpt-oss-120b", api_key="EMPTY")


def test_a_busy_server_is_retried_but_a_bad_request_is_not():
    """A local server has no rate limit; it can still be briefly unavailable."""
    assert is_transient_error(RuntimeError("503 Service Unavailable"))
    assert is_transient_error(ConnectionError("connection refused"))
    # A prompt longer than the context window will fail again just as fast.
    assert not is_transient_error(ValueError("maximum context length exceeded"))


# --- reading a reply ----------------------------------------------------------


def test_vllms_separated_reasoning_is_picked_up():
    reply = FakeReply('{"call": "resistant"}', {"reasoning": "rpoB S450L is tier 1"})
    assert reasoning_trace(reply) == "rpoB S450L is tier 1"


def test_the_older_reasoning_content_spelling_still_reads():
    reply = FakeReply("answer", {"reasoning_content": "thought"})
    assert reasoning_trace(reply) == "thought"


def test_reasoning_in_the_response_metadata_is_found_too():
    reply = FakeReply("answer", {}, {"reasoning": {"content": "thought"}})
    assert reasoning_trace(reply) == "thought"


def test_a_reply_without_reasoning_yields_an_empty_trace():
    assert reasoning_trace(FakeReply("answer")) == ""


def test_inline_think_tags_are_split_off_the_answer():
    answer, thinking = split_thinking(
        '<think>the rpoB variant is synonymous</think>{"call": "not_resistant"}'
    )
    assert answer == '{"call": "not_resistant"}'
    assert thinking == "the rpoB variant is synonymous"


def test_thinking_cut_off_by_a_token_limit_is_still_captured():
    answer, thinking = split_thinking("<think>I need to check katG")
    assert answer == ""
    assert thinking == "I need to check katG"


def test_content_blocks_are_joined():
    assert reply_text(FakeReply([{"type": "text", "text": "ab"}, "c"])) == "abc"


def test_answer_and_trace_prefers_the_servers_own_separation():
    reply = FakeReply("<think>draft</think>answer", {"reasoning": "the real trace"})
    assert answer_and_trace(reply) == ("answer", "the real trace")


# --- traces beside the predictions --------------------------------------------


def test_predict_keeps_the_trace_next_to_the_call(tmp_path):
    from tb_search.catalogue import panel_for
    from tb_search.cohort import Sample

    llm = FakeLLM(
        FakeReply(
            '{"call": "resistant", "variants_cited": [], "genes_cited": ["rpoB"]}',
            {"reasoning": "S450L sits in the rifampicin resistance hotspot"},
        )
    )
    sample = Sample(
        isolate="SRR1",
        drug="rifampicin",
        phenotype="R",
        vcf_path=str(tmp_path / "SRR1.vcf"),
    )
    prediction = predict(llm, sample, [], panel_for("rifampicin"))

    assert prediction["call"] == "resistant"
    assert prediction["reasoning_trace"].startswith("S450L sits in")
    assert RESISTANCE_PROMPT.split("\n")[0] in llm.prompts[0]


def test_a_captured_trace_is_measured_in_the_check():
    prediction = normalize_prediction({"call": "unknown"}, "forty two words of it")
    check = check_trace(prediction, [], panel_for_rifampicin())
    assert check.thinking_captured
    assert check.thinking_words == 5
    assert check.valid


def test_a_missing_trace_is_flagged_without_failing_the_call():
    check = check_trace(
        normalize_prediction({"call": "unknown"}), [], panel_for_rifampicin()
    )
    assert not check.thinking_captured
    assert check.thinking_words == 0
    # Nothing was invented, so the call still stands; the note says why there is
    # no thinking to read through.
    assert check.valid
    assert any("reasoning parser" in note for note in check.notes)


def panel_for_rifampicin():
    from tb_search.catalogue import panel_for

    return panel_for("rifampicin")


def test_the_run_reports_how_many_calls_came_with_thinking():
    from tb_search.evaluate import evaluate_resistance

    rows = [
        {
            "isolate": "A",
            "phenotype": "R",
            "call": "resistant",
            "trace": {"valid": True, "thinking_captured": True, "thinking_words": 100},
        },
        {
            "isolate": "B",
            "phenotype": "S",
            "call": "not_resistant",
            "trace": {"valid": True, "thinking_captured": False, "thinking_words": 0},
        },
    ]
    trace = evaluate_resistance(rows)["trace"]
    assert trace["thinking_captured"] == 0.5
    assert trace["median_thinking_words"] == 100
