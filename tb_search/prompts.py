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
