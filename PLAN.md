vcf files input → TB input (dataset) for each drug for big tb paper (present in unity)

inference is on groq now, not vllm on unity. biomni has a Groq source of its own,
so the A1 agent does its tool calling straight against api.groq.com and nothing
here holds weights or needs a card. the unity side is just a cpu job that runs
the client (scripts/run_unity_cpu.sbatch).
check medea and how it is → human readbale traces (it uses gpt-oss-120b under
the hood, which is also what we default to on groq)

do a binary predictions for them → resistant ,not resistant, unknown  ; the agent to take in the vcf files, for given drug what will be the reason for that

check the reasoning traces for them → for each of them -> capture the reasoning
traces from groq; the thinking comes back in its own `reasoning` field when the
request asks for it (include_reasoning for the gpt-oss models,
reasoning_format=parsed for the rest)
https://console.groq.com/docs/reasoning

what to watch instead of the gpu: the rate limit. every A1 step carries the
~35k token system prompt and is counted per minute, so logs/groq_usage.jsonl
records tokens, queue time and every second spent waiting out a 429
https://console.groq.com/docs/rate-limits

check their responses and how biologically valid the reasoning traces are →
