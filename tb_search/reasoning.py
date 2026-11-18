"""Pull a model's hidden reasoning out of a reply, whoever served it.

vLLM returns the thinking separately from the answer when the server runs with
`--reasoning-parser`: the chat-completion message then carries a `reasoning`
field (older vLLM called it `reasoning_content`) next to `content`.
langchain-openai documents that it does not extract such non-standard fields, so
tb_search.screening's ChatOpenAI subclass copies them onto the message's
`additional_kwargs`, which is where this module reads them. Ollama has no such
parser and leaves the thinking inline in <think> tags, so that case is handled
too.

The traces are the point of the run, not a side effect: a call is only as good
as the reasoning behind it, so the trace is captured for every prediction and
stored beside the variants the model was shown.
"""

import re

# vLLM's field, new spelling first.
REASONING_KEYS = ("reasoning", "reasoning_content")
# What vLLM accepts as `reasoning_effort`; "none" turns thinking off.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# A trace cut short by a token limit never gets its closing tag, hence the \Z.
THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>(.*?)(?:</\1>|\Z)", re.DOTALL | re.IGNORECASE
)


def normalize_effort(value: str | None) -> str | None:
    """Fold an effort to one vLLM accepts; None when nothing was asked for.

    Raises ValueError for anything vLLM would reject.
    """
    folded = str(value or "").strip().lower()
    if not folded:
        return None
    if folded not in EFFORTS:
        raise ValueError(
            f"unknown reasoning effort {value!r}; use one of: {', '.join(EFFORTS)}"
        )
    return folded


def split_thinking(text: str) -> tuple[str, str]:
    """(answer, thinking) for a reply that left its thinking inline in <think> tags."""
    thinking = [match.group(2).strip() for match in THINK_BLOCK.finditer(text)]
    answer = THINK_BLOCK.sub("", text).strip()
    return answer, "\n\n".join(part for part in thinking if part)


def reasoning_trace(reply) -> str:
    """The separated reasoning for one reply, or "" when the server sent none."""
    sent = getattr(reply, "additional_kwargs", None) or {}
    for key in REASONING_KEYS:
        trace = str(sent.get(key) or "").strip()
        if trace:
            return trace
    return ""


def answer_and_trace(reply) -> tuple[str, str]:
    """(answer text, reasoning trace) for one reply, from either shape.

    A server with a reasoning parser hands back the two already separated; one
    without it leaves <think> tags in the content, which would otherwise be fed
    to the JSON parser along with the answer.
    """
    answer, inline = split_thinking(reply.text)
    return answer, reasoning_trace(reply) or inline


def answer_text(reply) -> str:
    """The answer alone: a reasoning model's thinking is dropped, not parsed.

    A model served without a vLLM reasoning parser leaves its thinking inline in
    <think> tags, and that thinking often rehearses JSON it then discards — so
    feeding it to the verdict parser reads the draft instead of the answer.
    """
    return answer_and_trace(reply)[0]
