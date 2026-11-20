"""The Groq backend: settings, reasoning requests, rate-limit waits, captured traces."""

import pytest
from langchain_core.messages import AIMessage

from tb_search.config import DEFAULT_MODEL, ConfigError, Settings, load_settings
from tb_search.prompts import RESISTANCE_PROMPT
from tb_search.reasoning import (
    answer_and_trace,
    normalize_effort,
    reasoning_body,
    reasoning_trace,
    split_thinking,
)
from tb_search.resistance import check_trace, normalize_prediction, predict
from tb_search.screening import (
    apply_reasoning_body,
    backoff_seconds,
    failure_hint,
    is_context_error,
    is_rate_limit_error,
    is_token_budget_error,
    is_transient_error,
    reasoning_chat_model,
    retry_after_seconds,
    retry_transient,
    run_agent,
)

PROVIDER_ENV = (
    "GROQ_MODEL",
    "LLM_MODEL",
    "GROQ_BASE_URL",
    "LLM_BASE_URL",
    "GROQ_API_KEY",
    "LLM_SOURCE",
    "LLM_REASONING_EFFORT",
    "GROQ_REASONING_EFFORT",
    "GROQ_MAX_RETRIES",
)

TEST_KEY = "gsk_test_key"


@pytest.fixture
def clean_env(monkeypatch):
    """A bare environment with a key: .env on disk must not decide what these
    tests see, but every mode needs a key to get as far as the settings."""
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("tb_search.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GROQ_API_KEY", TEST_KEY)


def FakeReply(content, additional_kwargs=None):
    """What langchain hands back: content, plus whatever it did not recognise."""
    return AIMessage(content=content, additional_kwargs=additional_kwargs or {})


class FakeAgent:
    """An A1 stand-in: one canned (answer, thinking) per run.

    It mimics the shape run_agent depends on — `go` returning (log, content)
    and the `reasoning_traces` list build_agent attaches, which is where the
    real agent's thinking has to be collected because A1 drops it.
    """

    def __init__(self, *runs):
        self.runs = list(runs)
        self.prompts = []
        self.reasoning_traces: list[str] = []
        # build_agent sets this; without it the retries fall back to the default.
        self.max_retries = 0

    def go(self, prompt):
        self.prompts.append(prompt)
        content, thinking = self.runs.pop(0)
        if thinking:
            self.reasoning_traces.append(thinking)
        return ["step"], content


class FakeRateLimit(Exception):
    """A 429 as the OpenAI client raises it: a message, and the response headers."""

    def __init__(self, message, headers=None):
        super().__init__(message)
        self.response = type("Response", (), {"headers": headers or {}})()


# --- settings -----------------------------------------------------------------


def test_the_default_backend_is_groq(clean_env):
    settings = load_settings()
    assert settings.source == "Groq"
    assert settings.model == DEFAULT_MODEL == "openai/gpt-oss-120b"
    assert settings.base_url == "https://api.groq.com/openai/v1"
    assert settings.api_key == TEST_KEY
    # Named without the endpoint: there is only one, so it says nothing.
    assert "@" not in settings.description


def test_a_run_without_a_key_is_refused(clean_env, monkeypatch):
    """Unlike a model we served ourselves, Groq will not answer without one."""
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        load_settings()


def test_the_model_and_effort_can_come_from_the_environment(clean_env, monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "default")
    settings = load_settings()
    assert settings.model == "qwen/qwen3.8-27b"
    assert settings.reasoning_effort == "default"
    assert "effort default" in settings.description
    assert settings.label == "Groq/qwen/qwen3.8-27b"


def test_another_backend_is_refused(clean_env):
    """This pipeline runs on Groq; asking for anything else is a mistake."""
    for source in ("Custom", "Ollama", "OpenAI", "Anthropic"):
        with pytest.raises(ConfigError, match="not the backend"):
            load_settings(source=source)


def test_a_stale_source_in_the_environment_is_ignored(clean_env, monkeypatch, capsys):
    """This pipeline used to serve its own model; a leftover .env must not break it."""
    monkeypatch.setenv("LLM_SOURCE", "Custom")
    settings = load_settings()
    assert settings.source == "Groq"
    assert "ignored" in capsys.readouterr().out


def test_an_unknown_reasoning_effort_is_refused(clean_env):
    with pytest.raises(ConfigError):
        load_settings(reasoning_effort="maximum")


def test_the_retry_budget_comes_from_the_environment(clean_env, monkeypatch):
    monkeypatch.setenv("GROQ_MAX_RETRIES", "9")
    assert load_settings().max_retries == 9
    assert load_settings(max_retries=2).max_retries == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("high", "high"),
        (" NONE ", "none"),
        ("default", "default"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_effort(raw, expected):
    assert normalize_effort(raw) == expected


def test_normalize_effort_rejects_what_groq_would_not_take():
    for bad in ("very high", "minimal", "max"):
        with pytest.raises(ValueError):
            normalize_effort(bad)


# --- asking Groq to separate the thinking -------------------------------------


def test_gpt_oss_models_use_include_reasoning():
    """They reject reasoning_format outright and return reasoning by default."""
    body = reasoning_body("openai/gpt-oss-120b", "high")
    assert body == {"include_reasoning": True, "reasoning_effort": "high"}


def test_other_reasoning_models_ask_for_the_parsed_format():
    """`raw` is a 400 as soon as tool calling is on, which for A1 is always."""
    body = reasoning_body("qwen/qwen3.8-27b", "default")
    assert body == {"reasoning_format": "parsed", "reasoning_effort": "default"}


def test_the_two_reasoning_switches_are_never_sent_together():
    """Groq rejects a request carrying both."""
    for model in ("openai/gpt-oss-20b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"):
        body = reasoning_body(model)
        assert ("include_reasoning" in body) != ("reasoning_format" in body)
        assert "reasoning_effort" not in body


def test_the_reasoning_fields_ride_along_in_the_request_body(clean_env):
    settings = load_settings(reasoning_effort="high")
    chat = apply_reasoning_body(_chat_openai(), settings)
    assert chat.extra_body == {
        "include_reasoning": True,
        "reasoning_effort": "high",
    }
    # Still a chat model, not a bound runnable: A1 calls with_structured_output.
    assert hasattr(chat, "with_structured_output")


def test_without_an_effort_the_thinking_is_still_asked_for(clean_env):
    """No effort means Groq's own default for the model, not no reasoning."""
    chat = apply_reasoning_body(_chat_openai(), load_settings())
    assert chat.extra_body == {"include_reasoning": True}


def test_the_chat_model_keeps_the_reasoning_langchain_would_drop():
    """ChatOpenAI tracks the official schema and discards Groq's extra field."""
    chat = reasoning_chat_model(Settings(model="m", api_key="gsk_test"))
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


def test_the_other_field_name_is_kept_too():
    chat = reasoning_chat_model(Settings(model="m", api_key="gsk_test"))
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


def test_the_chat_model_does_not_retry_behind_our_back():
    """retry_transient waits out the 429s and says so; silent retries hide it."""
    assert (
        reasoning_chat_model(Settings(model="m", api_key="gsk_test")).max_retries == 0
    )


def _chat_openai():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=DEFAULT_MODEL, api_key="gsk_test")


# --- waiting out the rate limit -----------------------------------------------


def test_a_rate_limit_is_retried_but_a_bad_request_is_not():
    """The hosted endpoint is up; being over its per-minute limit is the failure."""
    assert is_rate_limit_error(FakeRateLimit("429 Rate limit reached for model"))
    assert is_transient_error(FakeRateLimit("429 Rate limit reached for model"))
    assert is_transient_error(RuntimeError("503 Service Unavailable"))
    assert is_transient_error(ConnectionError("connection reset"))
    # A prompt longer than the context window will fail again just as fast, and
    # so will a bad key or a model this account cannot reach.
    assert not is_transient_error(ValueError("maximum context length exceeded"))
    assert not is_transient_error(ValueError("401 Invalid API Key"))
    assert is_context_error(ValueError("maximum context length exceeded"))
    # "rate limit" messages name a limit in tokens, which must not read as a
    # context-length failure — that one is not retried.
    assert not is_context_error(
        FakeRateLimit("429 Rate limit reached: limit 6000 tokens per minute")
    )


def test_a_request_over_the_minutes_whole_budget_is_not_retried():
    """Groq's 413. It reads like a rate limit and waiting changes nothing: one
    A1 request carries ~36k tokens, which a free tier's 8,000 TPM cannot ever
    afford, so the run has to say that instead of backing off forever."""
    over = FakeRateLimit(
        "Error code: 413 - Request too large for model `openai/gpt-oss-120b` "
        "in organization `org_x` service tier `on_demand` on tokens per minute "
        "(TPM): Limit 8000, Requested 36071"
    )
    assert is_token_budget_error(over)
    assert not is_transient_error(over)
    # Nor a context-window failure: a shorter excerpt would not help either.
    assert not is_context_error(over)
    assert "tokens-per-minute" in failure_hint(over)
    assert "--tools" in failure_hint(over)


def test_the_hint_matches_the_failure():
    assert "context window" in failure_hint(ValueError("maximum context length"))
    assert failure_hint(FakeRateLimit("429 Rate limit reached")) is None


def test_groqs_own_retry_after_is_obeyed():
    """It comes in the header, and the message repeats it to the millisecond."""
    assert retry_after_seconds(FakeRateLimit("429", {"retry-after": "2"})) == 2
    assert retry_after_seconds(
        FakeRateLimit("Rate limit reached. Please try again in 7.481s")
    ) == pytest.approx(7.481)
    assert retry_after_seconds(
        FakeRateLimit("Please try again in 1.5m")
    ) == pytest.approx(90)
    assert retry_after_seconds(RuntimeError("503 Service Unavailable")) is None


def test_the_wait_follows_groq_when_it_said_and_backs_off_when_it_did_not():
    told = FakeRateLimit("429", {"retry-after": "6"})
    # A second of slack, so the window has actually rolled over by the retry.
    assert backoff_seconds(told, 1) == 7
    silent = RuntimeError("502 Bad Gateway")
    first, later = backoff_seconds(silent, 1), backoff_seconds(silent, 4)
    assert first < later <= 90


def test_a_rate_limited_call_is_waited_out_and_then_succeeds(monkeypatch):
    monkeypatch.setattr("tb_search.screening.time.sleep", lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise FakeRateLimit("429 Rate limit reached", {"retry-after": "1"})
        return "done"

    assert retry_transient(flaky, attempts=3) == "done"
    assert len(calls) == 3


def test_the_retries_run_out_and_the_error_is_raised(monkeypatch):
    """So the caller can mark one batch or isolate failed and carry on."""
    monkeypatch.setattr("tb_search.screening.time.sleep", lambda _s: None)
    calls = []

    def always_limited():
        calls.append(1)
        raise FakeRateLimit("429 Rate limit reached", {"retry-after": "1"})

    with pytest.raises(FakeRateLimit):
        retry_transient(always_limited, attempts=2)
    assert len(calls) == 3  # the first try, plus two retries


# --- reading a reply ----------------------------------------------------------


def test_groqs_separated_reasoning_is_picked_up():
    reply = FakeReply('{"call": "resistant"}', {"reasoning": "rpoB S450L is tier 1"})
    assert reasoning_trace(reply) == "rpoB S450L is tier 1"


def test_the_other_reasoning_content_spelling_still_reads():
    reply = FakeReply("answer", {"reasoning_content": "thought"})
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
    """langchain-core joins them; this only pins that we lean on it."""
    answer, _ = answer_and_trace(FakeReply([{"type": "text", "text": "ab"}, "c"]))
    assert answer == "abc"


def test_answer_and_trace_prefers_groqs_own_separation():
    reply = FakeReply("<think>draft</think>answer", {"reasoning": "the real trace"})
    assert answer_and_trace(reply) == ("answer", "the real trace")


# --- traces beside the predictions --------------------------------------------


def test_run_agent_collects_the_thinking_a1_would_have_dropped():
    """A1 rebuilds each step's message, so the trace only survives in the list."""
    agent = FakeAgent(("<solution>done</solution>", "step one; step two"))
    answer, trace = run_agent(agent, "go")
    assert answer == "<solution>done</solution>"
    assert trace == "step one; step two"


def test_run_agent_falls_back_to_inline_thinking():
    """A model that answers with its thinking inline leaves it in <think> tags."""
    agent = FakeAgent(("<think>weighing it up</think>the answer", ""))
    assert run_agent(agent, "go") == ("the answer", "weighing it up")


def test_each_run_starts_from_a_clean_trace():
    """One isolate's thinking must not be filed under the next one."""
    agent = FakeAgent(("a", "first"), ("b", "second"))
    assert run_agent(agent, "one")[1] == "first"
    assert run_agent(agent, "two")[1] == "second"


def test_predict_keeps_the_trace_next_to_the_call(tmp_path):
    from tb_search.catalogue import panel_for
    from tb_search.cohort import Sample

    agent = FakeAgent(
        (
            '<solution>{"call": "resistant", "genes_cited": ["rpoB"]}</solution>',
            "S450L sits in the rifampicin resistance hotspot",
        )
    )
    sample = Sample(
        isolate="SRR1",
        drug="rifampicin",
        phenotype="R",
        vcf_path=str(tmp_path / "SRR1.vcf"),
    )
    prediction = predict(agent, sample, [], panel_for("rifampicin"))

    assert prediction["call"] == "resistant"
    assert prediction["reasoning_trace"].startswith("S450L sits in")
    assert RESISTANCE_PROMPT.split("\n")[0] in agent.prompts[0]


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
    assert any("reasoning-effort" in note for note in check.notes)


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


# --- what each request cost ---------------------------------------------------


def test_each_completion_files_its_tokens_and_groqs_timings(tmp_path, monkeypatch):
    """Groq publishes no metrics endpoint, so the responses are the record."""
    import json

    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("TB_SEARCH_USAGE_LOG", str(log))
    chat = reasoning_chat_model(Settings(model="openai/gpt-oss-120b", api_key="gsk_t"))
    chat._create_chat_result(
        {
            "model": "openai/gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "a", "reasoning": "b"},
                }
            ],
            "usage": {
                "prompt_tokens": 36000,
                "completion_tokens": 420,
                "total_tokens": 36420,
                "completion_tokens_details": {"reasoning_tokens": 300},
            },
            "x_groq": {"usage": {"queue_time": 2.5, "total_time": 4.0}},
        }
    )

    event = json.loads(log.read_text().splitlines()[0])
    assert event["event"] == "completion"
    assert event["prompt_tokens"] == 36000
    assert event["reasoning_tokens"] == 300
    # The throttling signal, as opposed to the model simply generating more.
    assert event["queue_time"] == 2.5


def test_nothing_is_recorded_when_no_log_was_asked_for(monkeypatch):
    monkeypatch.delenv("TB_SEARCH_USAGE_LOG", raising=False)
    from tb_search.usage import record_completion

    record_completion({"usage": {"total_tokens": 1}})  # must not raise


def test_a_wait_is_recorded_as_well_as_a_request(tmp_path, monkeypatch):
    """A run that spent its walltime in backoff must not look like a slow model."""
    import json

    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("TB_SEARCH_USAGE_LOG", str(log))
    monkeypatch.setattr("tb_search.screening.time.sleep", lambda _s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise FakeRateLimit("429 Rate limit reached", {"retry-after": "3"})
        return "done"

    retry_transient(flaky, attempts=2)
    event = json.loads(log.read_text().splitlines()[0])
    assert event["event"] == "wait"
    assert event["reason"] == "rate limited"
    assert event["seconds"] == 4.0


# --- everything is an A1 run ---------------------------------------------------


# What biomni builds for itself and build_agent is expected to replace.
BIOMNI_OWN_MODEL = object()


class FakeA1:
    """Stands in for biomni.agent.A1: the attributes build_agent touches.

    `llm` is the plain chat model A1 builds for itself, and `configure` is the
    graph rebuild it offers after its tools change.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.llm = BIOMNI_OWN_MODEL
        self.configured = 0

    def configure(self):
        self.configured += 1
        # What the real graph does per step, which is the point of the swap.
        self.graph_llm = self.llm


@pytest.fixture
def fake_biomni(monkeypatch, clean_env):
    """build_agent against a stand-in A1, so the wiring can be asserted offline."""
    import sys
    import types

    built = {}

    def A1(**kwargs):
        built["agent"] = FakeA1(**kwargs)
        return built["agent"]

    agent_module = types.ModuleType("biomni.agent")
    agent_module.A1 = A1
    config_module = types.ModuleType("biomni.config")
    config_module.default_config = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "biomni.agent", agent_module)
    monkeypatch.setitem(sys.modules, "biomni.config", config_module)
    return built, config_module


def test_the_agent_is_the_only_thing_holding_a_model(fake_biomni, clean_env):
    """Every task is an A1 run, so the swapped model has to be the agent's own."""
    from tb_search.screening import build_agent

    _built, config_module = fake_biomni
    settings = load_settings(reasoning_effort="high")
    agent = build_agent(settings)

    # A1 was pointed at Groq, not left to auto-detect from the model id.
    assert agent.kwargs["source"] == "Groq"
    assert agent.kwargs["llm"] == DEFAULT_MODEL
    assert config_module.default_config.source == "Groq"
    # Biomni's own plain model is gone; ours, asking for the reasoning, is in.
    assert agent.llm is not BIOMNI_OWN_MODEL
    assert agent.llm.extra_body == {
        "include_reasoning": True,
        "reasoning_effort": "high",
    }
    # And the graph was rebuilt around it, so a biomni that captures its model
    # at configure time still calls ours.
    assert agent.configured == 1
    assert agent.graph_llm is agent.llm


def test_the_agent_carries_the_run_state_the_call_sites_read(fake_biomni, clean_env):
    """run_agent and retrieval read both of these off the agent itself."""
    from tb_search.screening import build_agent

    agent = build_agent(load_settings(max_retries=7))
    assert agent.reasoning_traces == []
    assert agent.max_retries == 7


def test_a_graph_rebuild_that_fails_does_not_lose_the_run(
    fake_biomni, clean_env, capsys
):
    """The swap already applies without it, so this is belt, not braces."""
    from tb_search.screening import build_agent

    def exploding_configure():
        raise RuntimeError("biomni changed shape")

    original = FakeA1.configure
    try:
        FakeA1.configure = lambda self: exploding_configure()
        agent = build_agent(load_settings())
    finally:
        FakeA1.configure = original
    assert agent.llm.extra_body == {"include_reasoning": True}
    assert "could not rebuild" in capsys.readouterr().out


def test_every_llm_mode_goes_through_the_agent():
    """No mode may reach the model except through A1's tool loop.

    The guard is on the source: a direct chat call added to a task path would
    show up here, which is the one thing this pipeline's design does not allow.
    """
    import pathlib

    allowed = {"screening.py": ["OpenAI(", "client.chat.completions"]}
    for path in sorted(pathlib.Path("tb_search").glob("*.py")):
        text = path.read_text()
        for marker in ("OpenAI(", "client.chat.completions", ".invoke("):
            if marker in text:
                assert marker in allowed.get(path.name, []), (
                    f"{path.name} reaches the model directly ({marker!r}); every "
                    "task is meant to be an agent.go run. If this is a new "
                    "preflight, allow it here on purpose."
                )
