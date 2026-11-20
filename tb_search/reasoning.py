"""Pull a model's hidden reasoning out of a reply, and ask Groq to separate it.

Groq returns the thinking separately from the answer when it is asked to: the
chat-completion message then carries a `reasoning` field next to `content`.
Which request field asks for that depends on the model family, and the two are
mutually exclusive, so `reasoning_body` below picks one:

- the GPT-OSS models (`openai/gpt-oss-*`) do not accept `reasoning_format` at
  all; they put their thinking in `reasoning` by default and `include_reasoning`
  is the switch for it.
- every other reasoning model takes `reasoning_format`, and `parsed` is the
  value that splits the thinking into its own field. `raw` leaves it inline in
  <think> tags and is a 400 as soon as tool calling or JSON mode is on — which
  for an A1 run is always — so it is never sent.

langchain-openai documents that it does not extract such non-standard fields, so
tb_search.screening's ChatOpenAI subclass copies them onto the message's
`additional_kwargs`, which is where this module reads them. A model asked for
`raw`, or one that ignores the request, leaves the thinking inline instead, so
that case is handled too.

A1 then rebuilds each step's reply as a fresh AIMessage, which drops
`additional_kwargs` again — so an agent run's traces are collected at the model
instead (`screening.run_agent`), and `split_thinking` here is the fallback.

The traces are the point of the run, not a side effect: a call is only as good
as the reasoning behind it, so the trace is captured for every prediction and
stored beside the variants the model was shown.
"""

import re

# Groq's field, with the spelling some OpenAI-compatible servers use second.
REASONING_KEYS = ("reasoning", "reasoning_content")
# What Groq accepts as `reasoning_effort`. Which of them a given model takes is
# the model's business: "none"/"default" are the Qwen thinking switch, while
# low/medium/high are the GPT-OSS scale and "xhigh" only exists on Muse Spark.
# Sending one the model does not know is a 400 from Groq, not a silent default.
EFFORTS = ("none", "default", "low", "medium", "high", "xhigh")
# The families that reject `reasoning_format` and use `include_reasoning`.
INCLUDE_REASONING_MODELS = ("gpt-oss",)

# A trace cut short by a token limit never gets its closing tag, hence the \Z.
THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>(.*?)(?:</\1>|\Z)", re.DOTALL | re.IGNORECASE
)


def normalize_effort(value: str | None) -> str | None:
    """Fold an effort to one Groq accepts; None when nothing was asked for.

    Raises ValueError for anything Groq would reject outright.
    """
    folded = str(value or "").strip().lower()
    if not folded:
        return None
    if folded not in EFFORTS:
        raise ValueError(
            f"unknown reasoning effort {value!r}; use one of: {', '.join(EFFORTS)}"
        )
    return folded


def reasoning_body(model: str, effort: str | None = None) -> dict:
    """The request fields that make Groq hand the thinking back separately.

    One of `include_reasoning` / `reasoning_format` only — Groq rejects both in
    the same request — chosen by model family, plus `reasoning_effort` when one
    was asked for.
    """
    name = str(model or "").lower()
    if any(family in name for family in INCLUDE_REASONING_MODELS):
        body = {"include_reasoning": True}
    else:
        body = {"reasoning_format": "parsed"}
    if effort:
        body["reasoning_effort"] = effort
    return body


def split_thinking(text: str) -> tuple[str, str]:
    """(answer, thinking) for a reply that left its thinking inline in <think> tags."""
    thinking = [match.group(2).strip() for match in THINK_BLOCK.finditer(text)]
    answer = THINK_BLOCK.sub("", text).strip()
    return answer, "\n\n".join(part for part in thinking if part)


def reasoning_trace(reply) -> str:
    """The separated reasoning for one reply, or "" when Groq sent none."""
    sent = getattr(reply, "additional_kwargs", None) or {}
    for key in REASONING_KEYS:
        trace = str(sent.get(key) or "").strip()
        if trace:
            return trace
    return ""


def answer_and_trace(reply) -> tuple[str, str]:
    """(answer text, reasoning trace) for one reply, from either shape.

    A reply that came back with its reasoning separated carries the two already
    apart; one that left it inline would otherwise feed the <think> block to the
    JSON parser along with the answer.
    """
    answer, inline = split_thinking(reply.text)
    return answer, reasoning_trace(reply) or inline
