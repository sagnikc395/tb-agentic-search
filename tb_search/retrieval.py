"""Run the A1 agent on a retrieval request and turn its reply into paper rows."""

import io
import re
from contextlib import redirect_stdout

from tb_search.prompts import RETRIEVAL_PROMPT
from tb_search.screening import _is_rate_limit_error, _wait_out_rate_limit
from tb_search.verdicts import parse_paper_list

RATE_LIMIT_HINT = (
    "    hint: A1's system prompt is huge; a free Groq tier will be rate limited"
)

# A1 calls its retrieval tools as python inside <execute> blocks, so the
# searches have to be read back out of the code it wrote; there is no tool
# event to listen for.
SEARCH_CALL = re.compile(r"\b(\w*(?:query|search|fetch|extract)\w*)\s*\(([^)]{0,200})")
STRING_ARG = re.compile(r"""["']([^"']{3,160})["']""")


def _searches(output: str) -> list[str]:
    """The retrieval calls in one agent step, as short console lines."""
    lines: list[str] = []
    for name, args in SEARCH_CALL.findall(output):
        argument = STRING_ARG.search(args)
        line = f"{name}({argument.group(1)})" if argument else f"{name}(...)"
        if line not in lines:
            lines.append(line)
    return lines


def _run_streamed(agent, prompt: str, verbose: bool) -> str:
    """Run the agent step by step, echoing its searches; returns the last message.

    biomni pretty-prints every step to stdout itself, which buries the run in
    walls of code, so that output is swallowed unless verbose is set and only
    the search trace is shown.
    """
    steps = agent.go_stream(prompt)
    answer = ""
    step_number = 0

    while True:
        try:
            if verbose:
                step = next(steps)
            else:
                with redirect_stdout(io.StringIO()):
                    step = next(steps)
        except StopIteration:
            break

        step_number += 1
        output = step.get("output") or ""
        answer = output or answer
        for line in _searches(output):
            print(f"  [{step_number}] {line}", flush=True)
        if "<solution>" in output:
            print(f"  [{step_number}] agent wrote up its answer", flush=True)

    return answer


def retrieve_papers(
    agent, query: str, max_papers: int = 10, verbose: bool = False
) -> list[dict]:
    """One agent run: query (often naming a seed paper by URL) -> paper rows.

    A failed run returns whatever the stricter retry recovered, else [] — an
    empty hit list must not take the whole CLI down.
    """
    prompt = RETRIEVAL_PROMPT.replace("{N}", str(max_papers)).replace("{QUERY}", query)

    try:
        answer = _wait_out_rate_limit(_run_streamed, agent, prompt, verbose)
    except Exception as e:  # noqa: BLE001 - a dead agent run must not stop the CLI
        print(f"  ! agent run failed: {e}", flush=True)
        if _is_rate_limit_error(e):
            print(RATE_LIMIT_HINT, flush=True)
        answer = ""

    rows = parse_paper_list(answer or "")
    if not rows:
        # One retry, with a sterner reminder.
        print("  no paper list in the reply; retrying once", flush=True)
        try:
            answer = _wait_out_rate_limit(
                _run_streamed,
                agent,
                prompt + "\n\nREMINDER: reply with ONLY the JSON list, no other text.",
                verbose,
            )
            rows = parse_paper_list(answer or "")
        except Exception as e:  # noqa: BLE001 - leave the run empty instead of crashing
            print(f"  ! retry failed, leaving the run empty: {e}", flush=True)

    return rows
