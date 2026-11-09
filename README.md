# NovaCell Plan Recommender

Fit-first plan and add-on recommendations for NovaCell (a made-up mobile operator). For each customer the recommender suggests up to three plan changes or add-ons that fit their actual usage, says **why** in plain language, shows **how the monthly bill changes**, and stays silent when the current plan already fits. An LLM writes the in-app message; code checks every message before it is shown.

> **All data is synthetic** (seed 7): 8,000 customers, 10 months of usage, 4,756 plan changes and add-on purchases. The catalogue matches the NovaCell plan catalogue used in the earlier support-assistant project.

- **Live app:** https://novacell-plan-recommender.vercel.app (no sign-in)

## The problem

NovaCell's in-app banner shows "Upgrade to Unlimited" to everyone, and agents follow a playbook: over the data cap → Unlimited; travelling → Day Pass. Customers are on the wrong plans in both directions. Some pay for Unlimited and use 4 GB; others pay $15 a day for roaming or $0.40 a minute for international calls when a pass or pack would cost less. Overpaying customers leave, and bill shock creates complaints.

## Which recommender (measured, not assumed)

Trained on the 2,159 choices made April–August 2026 and tested on the 698 made in September–October. Every method ranks the same eligible options for that customer.

| Method | In top 1 | In top 3 | NDCG@3 | Bill shock (top 1) | Value to customer (top 1) |
|---|---|---|---|---|---|
| Popularity (one banner for all) | 36% | 71% | 0.57 | 9% | +$4.13/mo |
| Today's agent playbook (rules) | 21% | 67% | 0.47 | 62% | −$24.88/mo |
| Item-to-item collaborative filtering | 42% | 76% | 0.62 | 3% | +$6.95/mo |
| Hybrid gradient-boosted ranker | 59% | 91% | 0.78 | 1.1% | +$16.76/mo |
| **Hybrid + fit-first guardrail (chosen)** | **58%** | **89%** | **0.77** | **0.1%** | **+$16.80/mo** |

*Bill shock* = the top suggestion raises the monthly bill by more than $15 without removing at least as much in charges or slowed data. *Value to customer* = bill change plus the value of data no longer slowed (assumed $1.50/GB).

- **When to show nothing:** suggestions are shown only when the customer is at least $3 a month better off. At the start of September that showed suggestions to 39% of customers. 14% of them changed plan in the next two months, against 9% for a banner shown to everyone, and the gate reached 63% of everyone who changed.
- **Fairness:** top-3 accuracy does not differ significantly by age band (p = 0.65), gender (p = 0.64) or region (p = 1.0). Gender and region are never inputs; age enters only as the 55+ plan's eligibility flag.
- **The business tension:** most good fits lower the bill. For the top suggestion, *fit first* changes the bill by −$18.46 a month on average; *balanced*, which sends downgrades of more than $15 to the retention team instead, changes it by −$4.23. Whether lower bills pay back in fewer cancellations and complaints is what the A/B test measures (PRD, section 11).

## How it works

```
monthly usage + holdings ─► eligible options (catalogue rules) ─► bill under each option ─► hybrid ranker ─► guardrail (no bill shock) ─► gate (≥ $3 better off) ─► top 3 + reasons ─► LLM message ─► code checks ─► shown
```

| Piece | File |
|---|---|
| Synthetic data | `ml/generate_data.py` → `data/` |
| Recommenders, evaluation, fairness, current recommendations | `ml/train.py` → `ml/results/metrics.json`, `public/model.json`, `public/customers.json`, `data/current_recs.json` |
| Message writer and checker | `lib/message.js`, `lib/llm.js` (Groq gpt-oss-120b → gpt-oss-20b → Gemma 4 31B on OpenRouter) |
| API | `api/message.js`, `api/health.js`; `api/evals.js` (off unless `EVALS_ENABLED=true` or a preview) |
| UI | `public/index.html` |
| Tests and evals | `npm test` (all template messages pass; broken messages caught), `scripts/build_evals.py`, `evals/` |

**Message checks:** every number must appear in the facts; only the recommended item is named; no pressure wording, promises or freebies; no talk of algorithms or tracking; no personal traits; and if the bill goes up the message must say so. A failed check gets one repair; then the template message is shown.

## Run locally

```bash
pip install numpy pandas scikit-learn scipy   # only to regenerate data and retrain
python3 ml/generate_data.py && python3 ml/train.py && python3 scripts/build_evals.py
npm test
cp .env.example .env    # add GROQ_API_KEY and OPENROUTER_API_KEY; never commit .env
node --env-file=.env scripts/dev_server.js
```

## Assumptions (synthetic)

Pay-as-you-go roaming $15 a day; international calls $0.40 a minute; slowed data is worth $1.50/GB and missing hotspot $1/GB to the customer; add-ons are priced for the one line that uses them. Moving from postpaid to prepaid is not offered in the app.
