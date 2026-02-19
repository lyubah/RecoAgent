# RecoAgent

**Intent-aware recommendation engine with transparent, deterministic scoring.**

Describe what you need in natural language. The system parses your intent, retrieves options from a curated catalog, ranks them with an auditable formula (not an LLM), and explains every decision. **The scoring model is the product:** weights are the main UI—change a slider and the ranking updates in real time.

---

## Try it

**[→ Live Demo](https://your-app-url.streamlit.app)** *(replace with your Streamlit Community Cloud URL after deployment)*

Use the sliders to adjust how much each dimension (price, integrations, ease of setup, rating, etc.) matters. The ranking re-sorts instantly—no extra API calls. The formula is the interface.

---

## What it does

You describe your problem:

> *"We're a 5-person fintech startup using Python and PostgreSQL. We need a monitoring tool that integrates with Slack, under $30/month per seat."*

The system:

1. **Parses** your intent into structured criteria (e.g. cost-sensitive, compliance-aware, simplicity-focused).
2. **Filters** the catalog by hard constraints (price, required integrations, team size).
3. **Retrieves** semantically similar tools from the filtered set (embedding search).
4. **Ranks** with a deterministic, auditable scoring formula—same input → same ranking every time.
5. **Explains** why each result matched, with score breakdowns and confidence.

Then **you take over:** sliders control the weights. Move "price fit" up and "ease of setup" down—the list re-ranks immediately. The recommendation view is just one view of the formula; the sliders are the product.

---

## Design decision: sliders as the product

Most recommenders hide how they rank. Here, **the scoring model is the interface.**

- Weights are not hidden—they are the main UI.
- Changing a slider re-sorts the ranking in real time (no LLM, no re-retrieval).
- The "product" is the transparent formula; recommendations are one view of it.

So you can see exactly why item A is above item B, and you can change the tradeoffs yourself.

---

## Architecture

- **One-shot pipeline (run once per query):** Intent parser (LLM) → Hard filter → Vector retrieval → Deterministic scorer → Explainer/validator. Result: a fixed set of candidates and an initial ranking.
- **Sliders (continuous):** User adjusts weight sliders. Re-score the same candidates with new weights and re-sort. No LLM, no API calls—just the formula.

```
User query → [Parse] → [Filter] → [Retrieve] → [Score] → Initial ranking + candidates in state
                              ↑
Sliders (weights) ────────────┴──→ Re-score(candidates, weights) → Updated ranking
```

**Tech:** LangGraph (orchestration), Claude (parse + explain), ChromaDB + embeddings (retrieval), pure Python/NumPy (scoring), Streamlit (UI).

---

## Project structure

```
recoagent/
├── README.md                 # This file
├── app.py                    # Streamlit UI (query + sliders + ranked list)
├── graph.py                  # LangGraph pipeline (parse → filter → retrieve → score → explain)
├── nodes/
│   ├── intent_parser.py      # LLM: natural language → structured JSON
│   ├── hard_filter.py        # Deterministic: constraint filtering
│   ├── retriever.py          # ChromaDB similarity search
│   ├── scorer.py             # Deterministic: weighted scoring formula
│   ├── explainer.py          # LLM: explanations + validation
│   └── re_retriever.py       # Relaxed re-retrieval when confidence low
├── models/
│   └── state.py             # Pydantic models + LangGraph state
├── config/
│   └── saas_tools.json      # Domain config (schema, weights, prompts)
├── data/
│   └── saas_tools.json      # Curated tool catalog
├── eval/
│   ├── test_queries.json    # Labeled test queries
│   └── run_eval.py          # Evaluation script
├── embeddings/
│   └── build_index.py       # Build ChromaDB index from catalog
└── requirements.txt
```

---

## Run locally

```bash
git clone https://github.com/lyubah/RecoAgent.git
cd RecoAgent

pip install -r requirements.txt

# Set API key (for intent parsing and explanations)
export ANTHROPIC_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

---

## Domain (demo)

The demo uses a curated catalog of SaaS/developer tools (monitoring, CI/CD, auth, databases, payments, analytics, etc.). The pipeline is domain-agnostic: swap the config and catalog to use another domain.

---

## Reference: core components

| Component        | Role |
|-----------------|------|
| **Intent parser** | LLM turns natural language into structured intent, preferences, hard constraints, and suggested scoring weights (used to set initial slider positions). |
| **Hard filter**   | Removes catalog items that violate hard constraints (e.g. max price, required integrations) before retrieval. |
| **Retriever**     | Embeds the structured intent and runs similarity search (e.g. ChromaDB) on the filtered catalog; returns top-K candidates. |
| **Scorer**        | Pure formula: weighted sum of normalized sub-scores (price fit, integration overlap, ease of setup, rating, feature match, similarity). Weights come from sliders (or initial defaults). |
| **Explainer**     | LLM generates short explanations and match quality; optional. Deterministic validation checks constraints. |
| **Sliders**       | UI controls for each weight. On change: re-score candidates with new weights and update the ranked list. |

---

## License

MIT (or your choice).
