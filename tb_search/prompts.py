SCREEN_DOC_PROMPT = """You are a literature screener for a tuberculosis research project.
Decide whether the document below is fundamentally about tuberculosis (TB) or
Mycobacterium tuberculosis.

TB-related means TB or M. tuberculosis is a central subject or a core component
of the work — e.g. diagnosing TB, its drug resistance, or a genomic/variant
database built around it. A passing mention of TB does not count. Judge by the
excerpt.

Reply with exactly one JSON object and nothing else:

{"tb_related": true, "confidence": "high", "reason": "short justification"}

tb_related must be a boolean, confidence one of high/medium/low, reason under 20
words. Mark borderline calls with confidence "low".

DOCUMENT ("{name}"):

---
{DOC}
---
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
