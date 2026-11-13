# Used when a document is screened on its own rather than against a seed paper.
DEFAULT_TOPIC = (
    "Tuberculosis and Mycobacterium tuberculosis research: drug resistance, "
    "genomics and variant interpretation, diagnostics, treatment."
)

ASSESS_PROMPT = """You are a literature screener for a tuberculosis research project.
Assess the document below against the TOPIC. Judge only by the text shown.

Reply with exactly one JSON object and nothing else:

{"relevance": 2, "tb_central": true, "paper_type": "method", "organism": "Mycobacterium tuberculosis", "drugs": ["rifampicin"], "genes_variants": ["rpoB S450L"], "methods": ["CNN"], "datasets": ["CRyPTIC"], "evidence": "verbatim quote from the text", "abstain": false, "reason": "short justification"}

- relevance: 3 = same question, method or dataset as the topic; 2 = clearly
  useful to it; 1 = tangential; 0 = unrelated.
- tb_central: true only when TB or M. tuberculosis is a central subject. A
  passing mention does not count.
- paper_type: one of method, benchmark, dataset, review, clinical, agent, other.
- drugs, genes_variants, methods, datasets: only items named in the text; use
  [] when there are none.
- evidence: one sentence copied word for word from the text that best supports
  your relevance score, under 40 words.
- abstain: true when the text is too thin to judge (e.g. a title with no
  abstract). Do not guess.
- reason: under 20 words.

TOPIC:

{TOPIC}

DOCUMENT ("{name}"):

---
{DOC}
---
"""

QUERY_PROMPT = """You are planning literature searches for a tuberculosis research project.

TOPIC:

{TOPIC}

Relevant papers found so far:
{FOUND}

Queries already used:
{USED}

Write {N} new search queries that would surface related papers the earlier
queries missed. Each query takes a different angle: the scientific question,
the method, the dataset, or a specific drug or gene. Each query is 2 to 5
plain keywords, with no boolean operators and no quotes. Do not repeat or
lightly reword a used query.

Reply with exactly one JSON list of strings and nothing else:

["keywords one", "keywords two"]
"""


BATCH_SCREEN_PROMPT = """You are a literature screener for a tuberculosis research project.
For each numbered document below, decide whether it is fundamentally about
tuberculosis (TB) or Mycobacterium tuberculosis.

A document is TB-related when TB or M. tuberculosis is a central subject or a
core component of the work — e.g. diagnosing TB, its drug resistance, or a
genomic/variant database built around it. A passing mention of TB does not
count. Judge each document by its excerpt.

Reply with exactly one JSON list, one object per document id, nothing outside it:

```json
[{"id": 0, "tb_related": true, "confidence": "high", "reason": "short justification"}]
```

Use a boolean for tb_related, one of high/medium/low for confidence, and keep
reasons under 20 words. Mark borderline calls with confidence "low".

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
