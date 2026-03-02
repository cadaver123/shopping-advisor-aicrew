# Shopping Advisor — CrewAI Multi-Agent System

A multi-agent shopping research assistant built with [CrewAI](https://github.com/crewAIInc/crewAI). Given a natural-language shopping query, six specialised agents work in parallel to gather expert reviews, Reddit community sentiment, and live marketplace prices from Allegro.pl and AliExpress, then synthesise everything into a ranked Markdown report.

## How it works

```
User query
    │
    ├── review_researcher   ──► Expert Review Search + Web Page Reader
    ├── reddit_researcher   ──► Reddit Community Search + Web Page Reader
    ├── allegro_researcher  ──► Allegro.pl Offer Search + Web Page Reader
    └── aliexpress_researcher──► AliExpress Offer Search + Web Page Reader
    │         (all four run in parallel)
    │
    ├── shopping_analyst    ──► Consolidates all findings, scores each product
    │
    └── shopping_advisor    ──► Produces the final Markdown report
```

All web searches go through [Serper.dev](https://serper.dev) (Google Search API). The LLM backend is [DeepSeek-V3](https://www.together.ai) via Together.ai.

## Prerequisites

- Python 3.11+
- A [Together.ai](https://www.together.ai) API key
- A [Serper.dev](https://serper.dev) API key

## Installation

```bash
git clone <repo-url>
cd crewai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# .env
TOGETHER_API_KEY=your_together_api_key
SERPER_API_KEY=your_serper_api_key

# Optional — defaults to deepseek-ai/DeepSeek-V3
TOGETHER_MODEL=deepseek-ai/DeepSeek-V3
```

## Usage

**Interactive mode:**
```bash
python main.py
```

**Query as a CLI argument:**
```bash
python main.py "Best noise-cancelling headphones under $250"
```

The final report is printed to the terminal and saved to `last_report.md`.

## Project structure

```
crewai/
├── main.py               # Entry point — query input, crew kickoff, output
├── crew.py               # Agent and task definitions (@CrewBase pattern)
├── tools.py              # Custom BaseTool implementations (search + web reader)
├── config/
│   ├── agents.yaml       # Agent roles, goals, and backstories
│   └── tasks.yaml        # Task descriptions and expected outputs
├── requirements.txt
└── last_report.md        # Auto-saved report from the last run
```

## Agents

| Agent | Role | Tools |
|---|---|---|
| `review_researcher` | Expert Review Researcher | Expert Review Search, Web Page Reader |
| `reddit_researcher` | Reddit Community Analyst | Reddit Community Search, Web Page Reader |
| `allegro_researcher` | Allegro.pl Price Scout | Allegro.pl Offer Search, Web Page Reader |
| `aliexpress_researcher` | AliExpress Deal Hunter | AliExpress Offer Search, Web Page Reader |
| `shopping_analyst` | Shopping Data Analyst | _(none — synthesises task outputs)_ |
| `shopping_advisor` | Personal Shopping Advisor | _(none — writes final report)_ |

## Report format

The final report is structured Markdown:

1. **Header** — restated query and one-sentence verdict
2. **Quick Picks** — summary table: Rank | Product | Score | Best For | Starting Price
3. **Detailed Reviews** — one section per product with pros, cons, and buy links
4. **Bottom Line** — which product to buy and why

## Debugging URL hallucinations

The LLM occasionally fabricates or mutates URLs. Three debug artefacts are written on every run:

| File | Contents |
|---|---|
| `tool_urls.log` | Every URL returned by Serper, tagged by tool name |
| `crew_run.log` | Full agent Thought/Action/Observation trace |
| terminal output | Per-task raw LLM outputs + live HTTP status check for every URL in the report |

**Workflow:**
1. Run the crew and note any broken links in the report.
2. Search `tool_urls.log` for that URL — if it is absent, the LLM invented it.
3. If it is present, open `crew_run.log` and search for the URL to find where it was mutated.
4. The terminal URL check (`[OK] 200` / `[BAD] 404`) gives an instant pass/fail for every link without opening a browser.
