# 🧬 Biomedical Literature Summarizer

Abstractive summarization of biomedical research papers, powered by
**Claude**. Designed to speed up systematic literature review by condensing
papers down to their key findings.

## Features

- **PDF upload** — extracts and summarizes the full text of a research paper PDF
- **Paste text** — summarize any abstract or passage directly
- **PubMed fetch** — pull the title + abstract for a PMID or PubMed URL via the
  NCBI E-utilities API and summarize it
- **Structured summary** — splits the summary into Abstract Summary / Materials
  and Methods / Results and Discussion sections
- **Reading-time stats** — shows estimated reading time for the original text
  vs. the summary, as a proxy for time saved during literature review

## Model

[`claude-sonnet-4-6`](https://www.anthropic.com/claude) via the Anthropic API.

## Project structure

| File | Purpose |
| --- | --- |
| `app.py` | Gradio UI and request handlers |
| `summarizer.py` | Claude API calls for plain and structured summarization |
| `document_utils.py` | PDF text extraction, text cleaning, PubMed fetching |
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

Open the printed `http://127.0.0.1:7860` URL in your browser.

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space with:
   - SDK: **Gradio**
   - Visibility: your choice
2. Add it as a git remote and push this repo:

```powershell
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

3. In the Space's **Settings → Variables and secrets**, add a secret named
   `ANTHROPIC_API_KEY` with your Anthropic API key.
4. The Space builds automatically and serves `app.py` at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

## Notes

- Each summary is a single Claude API call — billed per the
  [Anthropic API pricing](https://www.anthropic.com/pricing).
- PDF text extraction quality depends on how the PDF was generated — scanned
  (image-only) PDFs need OCR, which isn't included here.
