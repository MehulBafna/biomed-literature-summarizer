---
title: Biomedical Literature Summarizer
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Claude-based summarizer for biomedical research papers
---

# 🧬 Biomedical Literature Summarizer

**Live demo:** [huggingface.co/spaces/mehulbafna198/Medical_text_Summarizer](https://huggingface.co/spaces/mehulbafna198/Medical_text_Summarizer)

Abstractive summarization of biomedical research papers, powered by **Claude**. Designed to speed up systematic literature review.

## Features

- **PDF upload** — extracts and summarizes the full text of a research paper PDF
- **Paste text** — summarize any abstract or passage directly
- **PubMed fetch** — pull the title + abstract for a PMID or PubMed URL via the NCBI E-utilities API and summarize it
- **Keyword → Excel export** — search PubMed and Semantic Scholar by keyword, retrieve up to 15 relevant papers, summarize each with Claude, and download a structured Excel file (title, authors, key takeaway, methods, results)
- **Structured summary** — splits the summary into Abstract Summary / Materials and Methods / Results and Discussion sections
- **Biomedical scope guard** — rejects non-biomedical input with a clear message
- **Reading-time stats** — shows estimated reading time saved during literature review

## Model

[`claude-sonnet-4-6`](https://www.anthropic.com/claude) via the Anthropic API.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Gradio UI and request handlers |
| `summarizer.py` | Claude API calls for plain, structured, and batch export summarization |
| `document_utils.py` | PDF extraction, text cleaning, PubMed and Semantic Scholar search |
| `requirements.txt` | Python dependencies |

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root (gitignored) with:

```
ANTHROPIC_API_KEY=your-key-here
```

Then run:

```powershell
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

## Deploy to Hugging Face Spaces

1. Create a Space at https://huggingface.co/new-space (SDK: **Gradio**)
2. Add it as a git remote and push:

```powershell
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```

3. In **Settings → Variables and secrets**, add `ANTHROPIC_API_KEY`.

## Notes

- Each summary is a single Claude API call — billed per [Anthropic API pricing](https://www.anthropic.com/pricing).
- PDF text extraction requires text-based PDFs; scanned/image PDFs need OCR (not included).
