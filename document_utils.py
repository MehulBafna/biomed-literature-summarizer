"""Turn research-paper sources (PDF, raw text, PubMed) into clean plain text."""

import re
import time

import requests
from pypdf import PdfReader

PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_SS_HEADERS = {"User-Agent": "BiomedLiteratureSummarizer/1.0 (research tool)"}


def _get_with_retry(url: str, params: dict, timeout: int = 30, retries: int = 4) -> requests.Response:
    """GET with exponential backoff on 429 Too Many Requests."""
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429:
            wait = 2 ** attempt          # 1 s, 2 s, 4 s, 8 s
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


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
    response = _get_with_retry(PUBMED_EFETCH_URL, params=params)
    raw = response.text

    title = _medline_field(raw, "TI")
    abstract = _medline_field(raw, "AB")
    if not abstract:
        raise ValueError(f"No abstract found for PMID {pmid}.")

    combined = f"{title}\n\n{abstract}" if title else abstract
    return clean_text(combined)


# ---------------------------------------------------------------------------
# Keyword-based multi-paper search (for Excel export)
# ---------------------------------------------------------------------------

def _parse_authors(raw_authors: list[str], limit: int = 3) -> str:
    if not raw_authors:
        return "Unknown"
    display = raw_authors[:limit]
    suffix = " et al." if len(raw_authors) > limit else ""
    return ", ".join(display) + suffix


def search_pubmed_keywords(query: str, max_results: int = 10) -> list[dict]:
    """Search PubMed by keyword and return paper metadata dicts."""
    # Step 1: get PMIDs ranked by relevance
    esearch_resp = _get_with_retry(
        PUBMED_ESEARCH_URL,
        params={
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
        },
    )
    ids = esearch_resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    time.sleep(0.4)  # stay well within NCBI's 3 req/sec unauthenticated limit

    # Step 2: batch fetch full records in MEDLINE format
    efetch_resp = _get_with_retry(
        PUBMED_EFETCH_URL,
        params={"db": "pubmed", "id": ",".join(ids), "rettype": "medline", "retmode": "text"},
        timeout=30,
    )

    # Split on blank line before each new PMID- block
    records = re.split(r"\n(?=PMID- )", efetch_resp.text.strip())
    papers = []
    for rec in records:
        title = _medline_field(rec, "TI")
        abstract = _medline_field(rec, "AB")
        if not title or not abstract:
            continue
        pmid_val = _medline_field(rec, "PMID")
        authors = re.findall(r"^AU\s+-\s+(.+)$", rec, re.MULTILINE)
        year_m = re.search(r"^DP\s+-\s+(\d{4})", rec, re.MULTILINE)
        papers.append({
            "title": title,
            "authors": _parse_authors(authors),
            "year": year_m.group(1) if year_m else "",
            "abstract": clean_text(abstract),
            "source": "PubMed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_val}/" if pmid_val else "",
        })
    return papers


def search_semantic_scholar(query: str, max_results: int = 8) -> list[dict]:
    """Search Semantic Scholar by keyword and return paper metadata dicts."""
    try:
        resp = requests.get(
            SEMANTIC_SCHOLAR_URL,
            params={
                "query": query,
                "limit": max_results * 2,  # over-fetch; many lack abstracts
                "fields": "title,authors,abstract,year,externalIds",
            },
            headers=_SS_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
    except Exception:
        return []  # Semantic Scholar is best-effort; don't fail the whole search

    papers = []
    for p in resp.json().get("data", []):
        abstract = p.get("abstract") or ""
        title = p.get("title") or ""
        if not abstract or not title:
            continue
        authors = [a.get("name", "") for a in p.get("authors", [])]
        doi = (p.get("externalIds") or {}).get("DOI", "")
        papers.append({
            "title": title,
            "authors": _parse_authors(authors),
            "year": str(p.get("year") or ""),
            "abstract": clean_text(abstract),
            "source": "Semantic Scholar",
            "url": f"https://doi.org/{doi}" if doi else "",
        })
        if len(papers) >= max_results:
            break
    return papers


def _clean_query(query: str) -> str:
    """Normalise a user query for PubMed: replace punctuation separators with spaces."""
    q = re.sub(r"[;,/|()]+", " ", query)   # semicolons / commas / pipes → spaces
    q = re.sub(r"\s{2,}", " ", q)           # collapse runs of spaces
    return q.strip()


def _fallback_queries(query: str) -> list[str]:
    """Return progressively shorter versions of the query to try when the full query returns 0."""
    words = query.split()
    fallbacks = []
    # Drop one word at a time from the right until ≥2 words remain
    for end in range(len(words) - 1, 1, -1):
        fallbacks.append(" ".join(words[:end]))
    return fallbacks


def fetch_papers_for_export(query: str, total: int = 15) -> list[dict]:
    """Search PubMed (primary) + Semantic Scholar (supplement), deduplicate, return up to `total` papers."""
    base_query = _clean_query(query)

    # Try the cleaned query first; if empty, fall back to shorter versions
    queries_to_try = [base_query] + _fallback_queries(base_query)
    pubmed: list[dict] = []
    for q in queries_to_try:
        pubmed = search_pubmed_keywords(q, max_results=total)
        if pubmed:
            break   # found results — stop trying

    # Supplement with Semantic Scholar if PubMed came up short
    shortfall = total - len(pubmed)
    semantic = search_semantic_scholar(base_query, max_results=shortfall + 3) if shortfall > 0 else []

    seen = set()
    merged = []
    for paper in pubmed + semantic:
        key = re.sub(r"\W+", " ", paper["title"].lower()).strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(paper)
        if len(merged) >= total:
            break

    return merged
