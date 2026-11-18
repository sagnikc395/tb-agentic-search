"""Pull a model's hidden reasoning out of a reply, whoever served it.

vLLM returns the thinking separately from the answer when the server runs with
`--reasoning-parser`: the chat-completion message then carries a `reasoning`
field (older vLLM called it `reasoning_content`) next to `content`. langchain
keeps whatever it does not recognise in `additional_kwargs`, so that is where
both spellings land. Models served without a parser instead leave the thinking
inline in the content, wrapped in <think> tags, so that case is handled too.

The traces are the point of the run, not a side effect: a call is only as good
as the reasoning behind it, so the trace is captured for every prediction and
stored beside the variants the model was shown.
"""

import re

# vLLM's field, new spelling first. OpenAI-compatible servers that are not vLLM
# use one of the other two.
REASONING_KEYS = ("reasoning", "reasoning_content", "thinking")
# What vLLM accepts as `reasoning_effort`; "none" turns thinking off.
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>(.*?)</\1>", re.DOTALL | re.IGNORECASE
)
# A trace cut short by a token limit never gets its closing tag.
OPEN_THINK = re.compile(
    r"<(think|thinking|reasoning)>(.*)\Z", re.DOTALL | re.IGNORECASE
)


class EffortError(ValueError):
    """Raised when a reasoning effort is not one vLLM would accept."""


def normalize_effort(value: str | None) -> str | None:
    """Fold an effort to one vLLM accepts; None when nothing was asked for."""
    folded = str(value or "").strip().lower()
    if not folded:
        return None
    if folded not in EFFORTS:
        raise EffortError(
            f"unknown reasoning effort {value!r}; use one of: {', '.join(EFFORTS)}"
        )
    return folded


def split_thinking(text: str) -> tuple[str, str]:
    """(answer, thinking) for a reply that left its thinking inline in <think> tags."""
    if not text:
        return "", ""
    thinking = [match.group(2).strip() for match in THINK_BLOCK.finditer(text)]
    answer = THINK_BLOCK.sub("", text)
    unclosed = OPEN_THINK.search(answer)
    if unclosed is not None:
        thinking.append(unclosed.group(2).strip())
        answer = answer[: unclosed.start()]
    return answer.strip(), "\n\n".join(part for part in thinking if part)


def reply_text(reply) -> str:
    """The reply's content as one string, whether it came back as blocks or text."""
    content = getattr(reply, "content", reply)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            # langchain content blocks: {"type": "text", "text": ...}.
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            else:
                parts.append(str(block))
        return "".join(parts)
    return "" if content is None else str(content)


def _from_mapping(data) -> str:
    if not isinstance(data, dict):
        return ""
    for key in REASONING_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        # The Responses API nests it: {"reasoning": {"content": ...}}.
        if isinstance(value, dict):
            nested = _from_mapping(value) or str(value.get("content") or "").strip()
            if nested:
                return nested
    return ""


def reasoning_trace(reply) -> str:
    """The separated reasoning for one reply, or "" when the server sent none."""
    for holder in (
        getattr(reply, "additional_kwargs", None),
        getattr(reply, "response_metadata", None),
    ):
        found = _from_mapping(holder)
        if found:
            return found
    return ""


def answer_and_trace(reply) -> tuple[str, str]:
    """(answer text, reasoning trace) for one reply, from either shape.

    A server with a reasoning parser hands back the two already separated; one
    without it leaves <think> tags in the content, which would otherwise be fed
    to the JSON parser along with the answer.
    """
    text = reply_text(reply)
    trace = reasoning_trace(reply)
    answer, inline = split_thinking(text)
    return answer, trace or inline
