# Shopping Advisor

A pipeline that takes a natural-language shopping query in any language and returns community-vetted product recommendations with honest pros and cons — no marketing copy, no hallucinated prices.

## How it works

```
User query (any language)
    │
    ├── 1. DorkGenerator
    │       Detects language, category, market scope (global/local)
    │       Converts and buckets budget to USD ($20/$50/$100/$200/$500/$1000)
    │       Generates 3 optimised forum search queries in English or native language
    │
    ├── 2. SerperUrlCollector
    │       Runs queries through Serper.dev (Google Search API)
    │       Returns deduplicated forum/discussion URLs
    │
    ├── 3. ProductsExtractor
    │       Scrapes each URL via stealth Playwright browser
    │       LLM audits each candidate product:
    │         - INTERNAL PRICE CHECK: estimates MSRP from training data
    │         - HARD FILTER: discards anything over budget immediately
    │         - TYPE CHECK: discards wrong product types
    │       Returns up to 10 KEEP products with audit log
    │
    └── 4. RagEnricher
            Chunks scraped pages, embeds via multilingual-e5-large-instruct
            For each product: retrieves top-6 relevant chunks by cosine similarity
            LLM synthesises: pros, cons, street price, one-sentence verdict
```

## Tech stack

| Component | Technology |
|---|---|
| LLM (extraction, analysis) | DeepSeek-V3 via Together.ai |
| LLM (enrichment) | Llama-3.3-70B-Instruct-Turbo via Together.ai |
| Embeddings | `intfloat/multilingual-e5-large-instruct` via Together.ai (1024-dim) |
| Web search | Serper.dev (Google Search API) |
| Browser scraping | Playwright + playwright-stealth |
| Text extraction | trafilatura |
| DDG fallback search | ddgs |

## Prerequisites

- Python 3.11+
- A [Together.ai](https://www.together.ai) API key
- A [Serper.dev](https://serper.dev) API key
- Playwright browsers installed

## Installation

```bash
git clone <repo-url>
cd shopping-advisor-aicrew
pip install -r requirements.txt
playwright install chromium
```

## Configuration

```bash
cp .env.example .env
```

```dotenv
TOGETHER_API_KEY=your_together_api_key
SERPER_API_KEY=your_serper_api_key
```

## Usage

```bash
# Interactive prompt
python main.py

# Query as CLI argument
python main.py "Best noise-cancelling headphones under $250"
python main.py "Najlepsze słuchawki do 250zł"
python main.py "Beste Kopfhörer unter 200€"
```

## Project structure

```
├── main.py                        # Entry point — orchestrates all four phases
├── services/
│   ├── dork_generator.py          # Query generation + budget conversion/bucketing
│   ├── serper_url_collector.py    # Serper API → deduplicated forum URLs
│   ├── products_extractor.py      # Scrape + LLM budget auditor → product list
│   ├── rag_store.py               # Chunk + embed pages, cosine similarity search
│   └── rag_enricher.py            # Per-product RAG retrieval + pros/cons synthesis
├── tools/
│   ├── browser.py                 # Stealth Playwright worker thread
│   ├── embeddings.py              # Together.ai embeddings wrapper (batched, retried)
│   └── llm.py                     # Together.ai chat completions wrapper (with retry)
└── config/
    ├── models.yaml                # LLM model references (small/medium/strong)
    └── tools/
        ├── dork_generator.yaml    # Search query templates + bucketing tiers
        ├── products_extractor.yaml# Budget auditor system prompt + DATA AUDITOR schema
        └── rag_enricher.yaml      # Pros/cons synthesis prompt
```

## Budget filtering

The extractor applies a **zero-tolerance** budget policy. Every candidate product goes through a three-step audit logged to the terminal:

```
  ✓ [KEEP]    Moondrop Chu II ($19)   — MSRP under $50, matches type
  ✗ [DISCARD] Sony MDR7506 ($100)    — MSRP ($100) exceeds $50 budget
  ✗ [DISCARD] AKG K371 ($149)        — MSRP ($149) exceeds $50 budget
```

Budgets in foreign currencies are converted to USD and rounded **down** to the nearest standard tier before being passed to the LLM, so queries never contain irregular numbers like `$62`.

## Logging

Set `LOG_LEVEL=INFO` to see per-call LLM token usage, elapsed time, and embedding stats:

```
[LLM] ProductsExtractor     | DeepSeek-V3                   | 8420p + 312c = 8732 tok | 14.2s
[LLM] RagEnricher/Moondrop  | Llama-3.3-70B-Instruct-Turbo  |  980p +  98c = 1078 tok |  2.1s
[Embed] 65 texts | 3.4s
```
