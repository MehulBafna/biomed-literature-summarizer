"""Turn research-paper sources (PDF, raw text, PubMed) into clean plain text."""

import re

import requests
from pypdf import PdfReader

PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def get_filepath(file_obj):
    """Gradio File components may return a path string or a file-like object."""
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", file_obj)


def extract_text_from_pdf(file_obj) -> str:
    """Extract raw text from every page of a PDF."""
    reader = PdfReader(get_filepath(file_obj))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize whitespace and undo line-break hyphenation from PDF extraction."""
    if not text:
        return ""
    text = re.sub(r"-\s*\n\s*", "", text)  # re-join hyphen-broken words
    text = re.sub(r"\s*\n\s*", " ", text)  # collapse newlines
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse runs of spaces
    return text.strip()


def extract_pmid(pmid_or_url: str) -> str:
    """Pull a numeric PubMed ID out of a raw ID or a pubmed.ncbi.nlm.nih.gov URL."""
    match = re.search(r"(\d{4,9})", pmid_or_url.strip())
    if not match:
        raise ValueError("Could not find a PubMed ID in the input.")
    return match.group(1)


def _medline_field(raw: str, code: str) -> str:
    pattern = rf"^{code}\s*-\s*(.+(?:\n {{6}}.+)*)"
    match = re.search(pattern, raw, re.MULTILINE)
    if not match:
        return ""
    return re.sub(r"\n\s+", " ", match.group(1)).strip()


def fetch_pubmed_abstract(pmid_or_url: str) -> str:
    """Fetch the title + abstract for a PubMed ID via NCBI E-utilities."""
    pmid = extract_pmid(pmid_or_url)
    params = {"db": "pubmed", "id": pmid, "rettype": "medline", "retmode": "text"}
    response = requests.get(PUBMED_EFETCH_URL, params=params, timeout=15)
    response.raise_for_status()
    raw = response.text

    title = _medline_field(raw, "TI")
    abstract = _medline_field(raw, "AB")
    if not abstract:
        raise ValueError(f"No abstract found for PMID {pmid}.")

    combined = f"{title}\n\n{abstract}" if title else abstract
    return clean_text(combined)
