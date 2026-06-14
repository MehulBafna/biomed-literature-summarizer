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
