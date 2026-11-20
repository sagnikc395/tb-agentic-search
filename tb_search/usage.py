"""Record what every Groq request cost, as the run makes them.

On our own server the question "was this run slow because of the model or
because of us?" was answered by the card: utilisation, KV-cache occupancy,
tokens per second off vLLM's metrics endpoint. Groq publishes no such endpoint
to us — the only place that information exists is in the responses we are
already receiving, so this is where the run keeps it.

Every chat completion carries a `usage` block, and Groq adds its own timings
under `x_groq`: how long the request sat in the queue before it was served, and
how long the prefill and the decode took. Those three are what separate "the
account is being throttled" from "the model is thinking a lot per step", which
is the Groq-side version of the question above.

One JSON object per line, appended to the file named by TB_SEARCH_USAGE_LOG;
when that variable is unset, nothing is recorded and nothing is in the way.
run_experiment.sh sets it per run, scripts/monitor_run.sh samples it while the
run is going, and scripts/collect_results.sh totals it at the end.

Appending one short line per request under O_APPEND is atomic enough that the
sampler can read the file at the same time; a half-written last line is the
reader's problem to skip, which both of those scripts do.
"""

import json
import os
import time

LOG_ENV = "TB_SEARCH_USAGE_LOG"


def log_path() -> str:
    """Where usage is being recorded, or "" when it is not."""
    return (os.getenv(LOG_ENV) or "").strip()


def record(**fields) -> None:
    """Append one event. Never raises: a run must not die of its own bookkeeping."""
    path = log_path()
    if not path:
        return
    event = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
    try:
        with open(path, "a") as handle:
            handle.write(json.dumps(event) + "\n")
    except OSError:
        pass


def _number(value):
    """A usage field as a number, or None when the provider left it out."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def record_completion(response: dict, model: str = "") -> None:
    """File one chat completion's tokens and Groq's own timings for it."""
    if not log_path() or not isinstance(response, dict):
        return
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return
    details = usage.get("completion_tokens_details")
    groq = response.get("x_groq")
    timings = (groq or {}).get("usage") if isinstance(groq, dict) else None
    timings = timings if isinstance(timings, dict) else usage

    record(
        event="completion",
        model=response.get("model") or model,
        prompt_tokens=_number(usage.get("prompt_tokens")),
        completion_tokens=_number(usage.get("completion_tokens")),
        total_tokens=_number(usage.get("total_tokens")),
        # Present on the reasoning models: how much of the output was thinking
        # rather than answer, which is what --reasoning-effort actually buys.
        reasoning_tokens=_number((details or {}).get("reasoning_tokens"))
        if isinstance(details, dict)
        else None,
        # Groq's timings, in seconds. queue_time is the throttling signal: it
        # grows when the account is over its share, while completion_time grows
        # when the model is simply generating more.
        queue_time=_number(timings.get("queue_time")),
        prompt_time=_number(timings.get("prompt_time")),
        completion_time=_number(timings.get("completion_time")),
        total_time=_number(timings.get("total_time")),
    )


def record_wait(seconds: float, reason: str, attempt: int) -> None:
    """File a wait the run took rather than a request it made.

    A sweep that spends half its walltime in backoff looks, from the stage logs
    alone, exactly like one against a slow model. These rows are the difference.
    """
    record(
        event="wait", seconds=round(float(seconds), 2), reason=reason, attempt=attempt
    )
