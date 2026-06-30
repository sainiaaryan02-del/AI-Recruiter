# AI Candidate Ranking System
### Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Ranks 100,000 candidates for the **Senior AI Engineer — Founding Team** role at Redrob AI (Pune/Noida) using a 6-dimension scoring engine built directly from the job description.

---

## A Note on Scoring Scales

This system uses **two scales** for two different purposes — both shown throughout this document and the CSV:

| Scale | Used for | Where you'll see it |
|---|---|---|
| **0–100** | Internal composite score across the 6 weighted dimensions. Easiest for humans to read ("74/100"). | README, presentation deck, console reports |
| **0–1.0** | Final submission CSV `score` column, as required by `submission_spec.md` / `validate_submission.py`. | `output/submission.csv` only |

**Conversion:** `csv_score = (composite/100 × behavioral_multiplier) ÷ top_candidate's_value`, normalised so the #1 candidate = 1.0000 and all later ranks are strictly non-increasing, per the validator's requirements.

---

## How to Run

```bash
# No installation needed — pure Python standard library

# On the sample dataset
python rank.py --candidates data/sample_candidates.json --out output/submission.csv --top 50

# On the full 100K dataset (gzipped JSONL)
python rank.py --candidates candidates.jsonl.gz --out output/submission.csv --top 100
```

**Runtime:** ~18 seconds for 100K candidates on CPU. No GPU, no API calls.

---

## Scoring Architecture

```
Composite Score (0–100)  ×  Behavioral Multiplier (0.10–1.0)  =  Final Score
                                                                        │
                                                          normalised to 0–1.0 for CSV
```

### Composite Score — out of 100 (6 dimensions)

| Dimension | Weight | What it checks |
|---|---|---|
| Search & Retrieval Skills | **30 pts** | Pinecone, FAISS, Weaviate, Elasticsearch, Embeddings, Sentence Transformers |
| Career & Company Fit | **26 pts** | Product companies vs consulting; title trajectory; leadership |
| Applied ML Depth | **18 pts** | Pre-LLM era: XGBoost, LTR, BM25, PyTorch, fine-tuning |
| Experience Band | **12 pts** | 5–9 yr ideal; JD disqualifiers enforced |
| Location & Notice Period | **9 pts** | Pune/Noida; sub-30d notice; willing to relocate |
| Education | **5 pts** | Tier-1/2 institution; CS/AI/ML field |
| **Total** | **100 pts** | |

### Behavioral Multiplier — 0.10 to 1.0 (23 Redrob signals)

| Group | Weight | Key signals |
|---|---|---|
| Availability | 40% | open_to_work_flag, last_active_date, notice_period_days |
| Engagement Quality | 38% | recruiter_response_rate, avg_response_time_hours, interview_completion_rate |
| Profile Credibility | 22% | profile_completeness_score, verified_email, github_activity_score |

A candidate scoring 90/100 on paper but inactive for 6 months with a 5% response rate gets multiplied down to roughly 9–22 *effective* points — they look great but aren't reachable.

---

## Trap Detection

| Trap | Detection method |
|---|---|
| **Keyword Stuffers** | Check career history for actual shipping evidence, not skills list |
| **Ghost Candidates** | Score capped via exponential recency decay (half-life 14 days) on behavioral multiplier |
| **Consulting-Only** | Flag TCS/Infosys/Wipro/Accenture-only careers; composite hard-capped at 12/100 |
| **Wrong Domain** | Detect non-ML titles (HR Manager, Accountant) with no ML career evidence |
| **Honeypot Profiles** | Timeline check (YoE vs sum of career months); expert skill with 0 months used |

---

## Results on Sample Dataset (50 candidates)

| Rank | ID | Title | Composite /100 | Behavioral × | CSV Score (0–1) | Why |
|---|---|---|---|---|---|---|
| #1 | CAND_0000031 | Recommendation Systems Engineer | **74.4** | 0.558 | 1.0000 | FAISS, Pinecone, Embeddings, MLflow, BentoML · Swiggy/Uber/Zomato |
| #2 | CAND_0000038 | Java Developer | **38.9** | 0.467 | 0.4363 | Weaviate + MLOps · Swiggy/Hooli · Open to work |
| #3 | CAND_0000001 | Backend Engineer | **29.6** | 0.503 | 0.3579 | Fine-tuning LLMs, LoRA, W&B · Active job seeker |

- **7** potential fits
- **30** disqualified (consulting-only / wrong domain)
- **13** honeypots detected and removed

---

## Project Structure

```
ai-recruiter/
├── rank.py                      # Main ranking engine — run this
├── README.md
├── requirements.txt             # No dependencies needed
├── submission_metadata.yaml     # Hackathon metadata
│
├── data/
│   └── sample_candidates.json  # 50-candidate sample
│
└── output/
    └── submission.csv           # Ranked output (score column is 0–1.0)
```

---

## AI Tools

- Claude (Anthropic) — architecture discussion and code review
- No candidate data fed to any external LLM
- Ranking is fully deterministic — no API calls during ranking

---

*Redrob Intelligent Candidate Discovery & Ranking Challenge*
