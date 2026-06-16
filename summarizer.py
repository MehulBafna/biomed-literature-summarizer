"""Claude-based abstractive summarization of biomedical research papers."""

import os
import re

import anthropic

MODEL_NAME = "claude-sonnet-4-6"

# Generous character cap so a huge PDF can't blow up cost/latency; Sonnet's
# 200k-token context window comfortably covers a full paper well below this.
MAX_CHARS = 150_000

_NOT_FOUND = "Not addressed in the provided text."

_REFUSAL_TOKEN = "NOT_BIOMEDICAL"

_SCOPE_GUARD = (
    f"First, check whether the text below is a biomedical, medical, clinical, "
    f"or life-sciences research paper, abstract, preprint, or clinical study. "
    f"If it is NOT (e.g. it is a news article, story, code, general text, or "
    f"any other unrelated content), respond with exactly the single token "
    f"{_REFUSAL_TOKEN} and nothing else. Otherwise, ignore this check and "
    f"continue with the task below.\n\n"
)

_SCOPE_ERROR = (
    "This text doesn't appear to be a biomedical research paper, abstract, "
    "or clinical study. Please paste content related to biomedical or "
    "medical research."
)

_SECTION_HEADERS = (
    "ABSTRACT SUMMARY",
    "MATERIALS AND METHODS",
    "RESULTS AND DISCUSSION",
)

_SECTION_LABELS = {
    "ABSTRACT SUMMARY": "Abstract Summary",
    "MATERIALS AND METHODS": "Materials and Methods",
    "RESULTS AND DISCUSSION": "Results and Discussion",
}

_STRUCTURED_INSTRUCTIONS = f"""You are helping a researcher do a systematic literature review. Given the full text (or abstract) of a biomedical research paper, produce a structured summary with exactly these three sections:

1. {_SECTION_HEADERS[0]} (max 3-4 lines): combine background/context from the Abstract and Introduction with the study's objective.
2. {_SECTION_HEADERS[1]} (max 5 lines): summarize the study design, data, and methodology.
3. {_SECTION_HEADERS[2]} (max 5 lines): summarize the key findings, how they compare with other studies, and the conclusions.

Only use information present in the provided text - do not invent details, numbers, or study designs. If a section truly cannot be addressed from the text, write "{_NOT_FOUND}" for that section.

Respond with exactly these three headers, each followed by its summary, and nothing else:

{_SECTION_HEADERS[0]}:
...

{_SECTION_HEADERS[1]}:
...

{_SECTION_HEADERS[2]}:
..."""


def _client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it as an environment variable "
            "(or a Hugging Face Space secret) to enable summarization."
        )
    return anthropic.Anthropic(api_key=api_key)


def _truncate(text):
    return text[:MAX_CHARS]


def summarize_document(text, target_words=150, progress=None):
    """Summarize `text` as a single paragraph of roughly `target_words` words."""
    if progress:
        progress(0.5, desc="Generating summary...")
    client = _client()
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"{_SCOPE_GUARD}"
                f"Summarize the following biomedical research text in "
                f"approximately {target_words} words, focusing on the key "
                f"findings and conclusions. Respond with only the summary, "
                f"no preamble.\n\n{_truncate(text)}"
            ),
        }],
    )
    result = message.content[0].text.strip()
    if result == _REFUSAL_TOKEN:
        raise RuntimeError(_SCOPE_ERROR)
    return result


def summarize_structured(text, progress=None):
    """Summarize `text` as an Abstract / Materials & Methods / Results &
    Discussion breakdown, each condensed to a few lines."""
    if progress:
        progress(0.5, desc="Generating structured summary...")
    client = _client()
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"{_SCOPE_GUARD}{_STRUCTURED_INSTRUCTIONS}\n\nPAPER TEXT:\n{_truncate(text)}",
        }],
    )
    response_text = message.content[0].text.strip()
    if response_text == _REFUSAL_TOKEN:
        raise RuntimeError(_SCOPE_ERROR)
    return _parse_structured(response_text)


def _parse_structured(response_text):
    pattern = "|".join(re.escape(h) for h in _SECTION_HEADERS)
    parts = re.split(rf"(?:^|\n)({pattern}):[ \t]*\n?", response_text)
    result = {}
    for i in range(1, len(parts), 2):
        result[_SECTION_LABELS[parts[i]]] = parts[i + 1].strip()
    for header in _SECTION_HEADERS:
        result.setdefault(_SECTION_LABELS[header], _NOT_FOUND)
    return result


# ---------------------------------------------------------------------------
# Per-paper summarization for keyword → Excel export
# ---------------------------------------------------------------------------

_EXPORT_HEADERS = ("KEY TAKEAWAY", "ABSTRACT SUMMARY", "MATERIALS AND METHODS", "RESULTS")

_EXPORT_LABEL_MAP = {
    "KEY TAKEAWAY": "key_takeaway",
    "ABSTRACT SUMMARY": "abstract_summary",
    "MATERIALS AND METHODS": "materials_methods",
    "RESULTS": "results",
}

_EXPORT_INSTRUCTIONS = f"""You are extracting structured information from a biomedical research paper abstract for a systematic literature review spreadsheet.

Given the title and abstract below, produce exactly these four fields:

{_EXPORT_HEADERS[0]} (1 sentence): The single most important contribution or finding of this paper.
{_EXPORT_HEADERS[1]} (2-3 sentences): Concise summary of the full abstract including background and objective.
{_EXPORT_HEADERS[2]} (1-2 sentences): Study design, dataset, and key methodology. Write "{_NOT_FOUND}" if not described in the abstract.
{_EXPORT_HEADERS[3]} (1-2 sentences): Main quantitative or qualitative results and conclusions. Write "{_NOT_FOUND}" if not described in the abstract.

Use only information present in the provided text. Do not invent numbers, study designs, or outcomes.

Respond with exactly these four headers each followed by their content and nothing else:

{_EXPORT_HEADERS[0]}:
...

{_EXPORT_HEADERS[1]}:
...

{_EXPORT_HEADERS[2]}:
...

{_EXPORT_HEADERS[3]}:
..."""


def summarize_paper_for_export(paper: dict) -> dict:
    """Run Claude on one paper's title+abstract; return the paper dict enriched with summary fields."""
    client = _client()
    content = f"TITLE: {paper['title']}\n\nABSTRACT: {paper['abstract']}"
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"{_EXPORT_INSTRUCTIONS}\n\n{content}",
        }],
    )
    parsed = _parse_export_summary(message.content[0].text)
    return {**paper, **parsed}


def _parse_export_summary(text: str) -> dict:
    pattern = "|".join(re.escape(h) for h in _EXPORT_HEADERS)
    parts = re.split(rf"(?:^|\n)({pattern}):\s*\n?", text)
    result = {}
    for i in range(1, len(parts), 2):
        key = _EXPORT_LABEL_MAP.get(parts[i], parts[i].lower())
        result[key] = parts[i + 1].strip() if i + 1 < len(parts) else _NOT_FOUND
    for key in _EXPORT_LABEL_MAP.values():
        result.setdefault(key, _NOT_FOUND)
    return result
