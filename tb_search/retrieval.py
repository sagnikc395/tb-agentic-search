"""Run the A1 agent on a retrieval request and turn its reply into paper rows."""

from tb_search.prompts import RETRIEVAL_PROMPT
from tb_search.screening import _is_rate_limit_error, _wait_out_rate_limit
from tb_search.verdicts import parse_paper_list

RATE_LIMIT_HINT = (
    "    hint: A1's system prompt is huge; a free Groq tier will be rate limited"
)


def retrieve_papers(agent, query: str, max_papers: int = 10) -> list[dict]:
    """One agent run: query (often naming a seed paper by URL) -> paper rows.

    A failed run returns whatever the stricter retry recovered, else [] — an
    empty hit list must not take the whole CLI down.
    """
    prompt = RETRIEVAL_PROMPT.replace("{N}", str(max_papers)).replace("{QUERY}", query)

    try:
        _log, answer = _wait_out_rate_limit(agent.go, prompt)
    except Exception as e:  # noqa: BLE001 - a dead agent run must not stop the CLI
        print(f"  ! agent run failed: {e}", flush=True)
        if _is_rate_limit_error(e):
            print(RATE_LIMIT_HINT, flush=True)
        answer = ""

    rows = parse_paper_list(answer or "")
    if not rows:
        # One retry, with a sterner reminder.
        try:
            _log, answer = _wait_out_rate_limit(
                agent.go, prompt + "\n\nREMINDER: reply with ONLY the JSON list, no other text."
            )
            rows = parse_paper_list(answer or "")
        except Exception as e:  # noqa: BLE001 - leave the run empty instead of crashing
            print(f"  ! retry failed, leaving the run empty: {e}", flush=True)

    print(f"  found {len(rows)} paper(s)", flush=True)
    return rows
