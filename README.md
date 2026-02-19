# RecoAgent

**Intent-aware recommendation engine with transparent, deterministic scoring.**

Describe what you need in natural language. The system parses your intent, retrieves options from a curated catalog, and ranks them with a **quantum-inspired** slate optimizer: Born-rule relevance (match probability) + interference penalty (diversity). No actual quantum hardware—just the Hilbert-space view, implemented as real math. **The scoring model is the product:** sliders control the tradeoffs (e.g. diversity strength); the ranking updates in real time.

**Sure, it *looks* like a generic search engine—you type stuff, you get a list. But we threw that playbook in the shredder and used the power of quantum-inspired slate optimization instead.** Born-rule match probabilities! Interference so your top picks don’t all look the same! A slate that’s actually curated, explainable, and (we’ll say it) kind of fancy. No black box. Just math that decided to dress up as physics for the day.

---

## Try it

**[→ Live Demo](https://your-app-url.streamlit.app)** *(replace with your Streamlit Community Cloud URL after deployment)*

Use the sliders to adjust match vs diversity (and optional constraint softness). The ranking re-sorts instantly—no extra API calls. You see "match probability" and "interference added" per item; the formula is the interface.

---

## What it does

You describe your problem:

> *"We're a 5-person fintech startup using Python and PostgreSQL. We need a monitoring tool that integrates with Slack, under $30/month per seat."*

The system:

1. **Parses** your intent into structured criteria (e.g. cost-sensitive, compliance-aware, simplicity-focused).
2. **Filters** the catalog by hard constraints (price, required integrations, team size).
3. **Retrieves** semantically similar tools from the filtered set (embedding search).
4. **Ranks** with a quantum-inspired slate optimizer (Born-rule relevance + interference for diversity)—deterministic, same input → same slate every time.
5. **Explains** why each result matched: match probability, interference added, and "selected because" (high match, low redundancy).

Then **you take over:** sliders control the weights. Move "price fit" up and "ease of setup" down—the list re-ranks immediately. The recommendation view is just one view of the formula; the sliders are the product.

---

## Design decision: sliders as the product

Most recommenders hide how they rank. Here, **the scoring model is the interface.**

- Weights are not hidden—they are the main UI.
- Changing a slider re-sorts the ranking in real time (no LLM, no re-retrieval).
- The "product" is the transparent formula; recommendations are one view of it.

So you can see exactly why item A is above item B, and you can change the tradeoffs yourself.

---

## Quantum-inspired scoring (the twist)

We use **quantum-inspired geometry**—no complex numbers, no quantum hardware. Just the Hilbert-space view: items and query as normalized vectors, **Born-rule relevance** (probability of match), and an **interference penalty** so the slate isn’t redundant. It’s mathematically clean, easy to code, and visibly different from a plain weighted sum. Legit to ML engineers; cute + radical without cosplay.

### Why this works

- Plugs into your current pipeline: retriever → **slate optimizer** (replaces classic scorer) → explainer.
- Deterministic and explainable (every term is explicit).
- No training data required (unlike preference learning / bandits).
- Demo pops: you can show "match probability" and "interference" per item.

### The model

**1) Normalized vectors**  
Query and each item are embedding vectors (e.g. sentence-transformers), normalized: $|\phi|=1$, $|\psi_x|=1$.

**2) Born-rule relevance (match probability)**  
$$r(x) = |\langle \phi, \psi_x \rangle|^2$$  
For real normalized vectors this is **cosine similarity squared**.

**3) Interference penalty (redundancy)**  
Pairwise overlap over the slate $S$:
$$I(S) = \sum_{i<j} |\langle \psi_{x_i}, \psi_{x_j} \rangle|^2$$

**4) Constraints (optional soft barrier)**  
Hard filter stays. For soft relaxation:
$$B(x) = \sum_j \lambda_j \, \max(0, g_j(x))^2$$

**5) Slate score (maximize)**  
$$J(S) = \sum_{x \in S} r(x) - \gamma\, I(S) - \beta \sum_{x \in S} B(x)$$  
Pick $K$ items by **greedy selection** (fast, stable).

### Where it plugs in

- **Keep:** Intent parser, hard filter, retriever.
- **Replace:** The old "scorer" with a **slate optimizer** that:
  1. Computes $r(x)$ for top-$N$ candidates.
  2. Builds the slate by greedily maximizing $J(S)$.
  3. Stores per-item contributions + interference for explanations.

### What the UI shows

For each recommended item:

- **Match probability** = $r(x)$
- **Interference added** = how much redundancy it introduces vs already-selected items
- **Selected because** = high match probability with low interference

### Defaults (stable, not flaky)

| Parameter | Value |
|-----------|--------|
| Candidate pool $N$ (after filter + retrieval) | 30 |
| Slate size $K$ | 5 |
| Diversity strength $\gamma$ | 0.2–0.6 (slider) |
| Constraints | Hard filter first; barrier $\beta$ only if you want soft failure |

### Optional upgrade: mixed-state preference

When the query is ambiguous, use **multiple intent vectors** $\phi_1, \phi_2, \phi_3$ (e.g. "cheap", "enterprise", "easy setup") with LLM-derived weights $p_k$:
$$r(x) = \sum_k p_k \, |\langle \phi_k, \psi_x \rangle|^2$$  
Single formula; "superposition" vibes without extra complexity.

---

## Architecture

- **One-shot pipeline (run once per query):** Intent parser (LLM) → Hard filter → Vector retrieval → **Slate optimizer** (Born-rule + interference) → Explainer. Result: a fixed set of candidates and an initial slate of $K$ items.
- **Sliders (continuous):** User adjusts diversity strength $\gamma$ (and optional $\beta$). Re-run slate optimization on the same candidates; slate updates in real time. No LLM, no API calls—just the formula.

```
User query → [Parse] → [Filter] → [Retrieve] → [Slate optimizer] → Initial slate + candidates in state
                                    ↑
Sliders (γ, β) ────────────────────┴──→ Re-optimize(candidates, γ, β) → Updated slate
```

**Tech:** LangGraph (orchestration), Claude (parse + explain), ChromaDB + embeddings (retrieval), quantum-inspired slate optimizer (pure Python/NumPy), Streamlit (UI).

---

## Project structure

```
recoagent/
├── README.md                 # This file
├── app.py                    # Streamlit UI (query + sliders + ranked list)
├── graph.py                  # LangGraph pipeline (parse → filter → retrieve → slate optimize → explain)
├── nodes/
│   ├── intent_parser.py      # LLM: natural language → structured JSON
│   ├── hard_filter.py        # Deterministic: constraint filtering
│   ├── retriever.py          # ChromaDB similarity search
│   ├── scorer.py             # Slate optimizer: Born-rule + interference (replaces weighted sum)
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
| **Slate optimizer** | Quantum-inspired: Born-rule relevance $r(x)$ (cosine²) + interference penalty $I(S)$ for diversity. Greedy slate construction to maximize $J(S)$. Sliders control $\gamma$ (diversity) and optional $\beta$ (constraint softness). |
| **Explainer**     | LLM generates short explanations and match quality; optional. Deterministic validation checks constraints. |
| **Sliders**       | UI controls for each weight. On change: re-score candidates with new weights and update the ranked list. |

---

## License

MIT (or your choice).
