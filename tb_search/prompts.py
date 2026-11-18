"""The task prompts handed to the A1 agent.

Biomni's A1 brings its own ~35k-token system prompt — who it is, the tool
catalogue, the <execute>/<solution> protocol — but no task prompts, because
A1's design is that the caller supplies the task in `agent.go(...)`. These are
those tasks: what to judge, the domain rules to judge it by, and the JSON shape
the reply has to come back in so it can be parsed, normalized and checked
rather than taken on trust.
"""

BATCH_SCREEN_PROMPT = """You are a literature screener for a tuberculosis research project.
For each numbered document below, decide whether it is fundamentally about
tuberculosis (TB) or Mycobacterium tuberculosis.

A document is TB-related when TB or M. tuberculosis is a central subject or a
core component of the work — e.g. diagnosing TB, its drug resistance, or a
genomic/variant database built around it. A passing mention of TB does not
count. Judge each document by its excerpt; do not search for the paper.

Reply with exactly one JSON list, one object per document id, nothing outside it:

```json
[{"id": 0, "tb_related": true, "confidence": "high", "evidence": "verbatim quote from the excerpt", "reason": "short justification"}]
```

- tb_related: a boolean.
- confidence: one of high/medium/low. Mark borderline calls "low".
- evidence: one sentence copied word for word from that document's excerpt that
  best supports your verdict, under 40 words. Use "" if the excerpt is too thin
  to quote.
- reason: under 20 words.

DOCUMENTS:

{DOCS}
"""

RETRIEVAL_PROMPT = """You are a literature-retrieval agent for a tuberculosis research project.
The request below asks you to find papers relevant to a query. The query usually
names one seed paper by URL or DOI.

Steps:

1. If the query points at a seed paper, fetch it (extract_url_content for a
   webpage, query_pubmed/query_arxiv/query_scholar for a DOI or title) and note
   its title, abstract and methods.
2. Use query_pubmed, query_arxiv, query_scholar and search_google to find at
   most {N} papers closely related to the query: same question, same method,
   or the same dataset. Only include papers you confirmed through an actual
   search result — never invent titles.
3. Reply with exactly one JSON list, one object per paper, and nothing outside
   it:

```json
[{"title": "...", "year": "...", "link": "...", "source": "e.g. pubmed, arxiv, crossref", "relevance": "one sentence on how it relates to the query or seed paper"}]
```

Keep each relevance sentence under 25 words; for the seed paper itself, use
"seed" as the source.

REQUEST:

{QUERY}
"""


# Used for one isolate / one drug. The variant list is pre-filtered to the
# drug's panel genes and numbered, so a cited variant can be checked against
# what was actually in the VCF instead of taken on trust.
RESISTANCE_PROMPT = """You are a tuberculosis molecular-resistance analyst.
Given the variants called in one Mycobacterium tuberculosis isolate, predict
whether it is phenotypically resistant to {DRUG}.

Reason only from the variants listed — everything you need is in this prompt,
so do not search or run code to look anything up. Genomic coordinates are H37Rv
(NC_000962.3). Codon numbers and promoter offsets are already resolved for you.

Genes graded for {DRUG}:
- tier 1 (established mechanism): {TIER1}
- tier 2 (secondary): {TIER2}

Reply with exactly one JSON object and nothing else:

{"call": "resistant", "confidence": "high", "variants_cited": [0], "genes_cited": ["rpoB"], "mechanism": "how the variant causes resistance", "reasoning": "step-by-step justification", "unknown_reason": ""}

- call: one of "resistant", "not_resistant", "unknown". Use "unknown" only when
  the variants genuinely do not settle it — not as a hedge.
- confidence: one of high, medium, low.
- variants_cited: the ids of the listed variants your call rests on, as
  integers. Use [] when no variant supports the call. Never cite an id that is
  not in the list.
- genes_cited: the genes your reasoning rests on, named as above.
- mechanism: how the cited variants change the drug's target or activation,
  under 40 words. Leave "" when you cite no variant.
- reasoning: your reasoning, under 120 words. Name the variant, the gene, and
  why it does or does not confer resistance to {DRUG}. If a known resistance
  gene carries no variant, say so — an absence is evidence too.
- unknown_reason: under 25 words, filled in only when call is "unknown".

Weigh the evidence as follows:

- A variant labelled "synonymous" does not change the protein and is not
  resistance evidence, even in a tier 1 gene and even in a hotspot codon.
- A truncating variant (frameshift, or a change labelled "nonsense") in a gene
  whose loss of function causes resistance does count as resistance evidence.
- "AF" is the fraction of reads carrying the variant. A variant present in only
  a minority of reads is weak evidence on its own: it may be a subpopulation or
  a sequencing artefact. Say so, and lower your confidence, rather than reading
  it as a clear resistant call.
- A variant in a gene unrelated to {DRUG} is not evidence either way.

ISOLATE {ISOLATE} — variants in the {DRUG} panel genes ({COUNT} found):

---
{VARIANTS}
---
"""
