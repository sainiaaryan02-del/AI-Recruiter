# AI Candidate Discovery & Ranking System
### Redrob Hackathon for Intelligent Candidate Discovery & Ranking

> Ranks candidates the way a great recruiter would not by matching keywords, but by understanding who genuinely fits.

---

## The Role
**Senior AI Engineer — Founding Team Redrob AI · Pune / Noida, India**

---

## What This Does

Reads candidate profiles and ranks them for the above role using 6 weighted dimensions pulled directly from the job description:

| Dimension | Weight | What it measures |
|---|---|---|
| Search & Retrieval Skills | 30% | Pinecone, FAISS, Weaviate, Elasticsearch, Embeddings, Sentence Transformers |
| Career & Company Fit | 26% | Product companies vs consulting; no title-chasers; ownership language |
| Applied ML Depth (Pre-LLM) | 18% | XGBoost, LTR, BM25, PyTorch, fine-tuning — real depth |
| Experience Band | 12% | 5–9 yr ideal; JD disqualifiers enforced |
| Location & Notice Period | 9% | Pune/Noida preferred; sub-30d notice ideal |
| Education | 5% | Tier-1/2 institution; CS/AI/ML field |

**Final score = Composite score × Behavioral multiplier (0.10–1.0)**

The behavioral multiplier is built from 23 Redrob platform signals (open-to-work, last active date, recruiter response rate, notice period, interview completion rate, GitHub activity, profile completeness, and more).

---

## Trap Detection

The dataset contains built-in traps. We handle all of them:

- **Keyword Stuffers** — Lists AI skills but career history is HR/Marketing. We check job descriptions for actual shipping evidence.
- **Ghost Candidates** — Great CV but inactive 6+ months, <10% response rate. Behavioral multiplier down-weights to 0.1–0.25×.
- **Consulting-Only CVs** — Entire career at TCS/Infosys/Wipro/Accenture. JD explicitly disqualifies these. Score capped at 12/100.
- **Honeypot Profiles** — Impossible timelines (8 yrs at 3-yr-old company) or expert skills with 0 months used. Flagged before ranking.

---

## Quick Start

```bash
# No installation needed — pure Python standard library

# Run on the sample dataset (50 candidates)
python rank.py --candidates data/sample_candidates.json --out output/submission.csv --top 50

# Run on the full dataset (100K candidates → top 100)
python rank.py --candidates candidates.jsonl --out output/submission.csv --top 100

# With gzipped JSONL
python rank.py --candidates candidates.jsonl.gz --out output/submission.csv --top 100
```

**Runtime:** ~18 seconds for 100K candidates on CPU. Well within the 5-minute constraint.

---

## Project Structure

```
ai-recruiter/
├── rank.py                        # Main ranking engine (run this)
├── requirements.txt               # No dependencies needed
├── README.md
├── submission_metadata.yaml       # Hackathon submission metadata
│
├── data/
│   ├── sample_candidates.json     # 50-candidate sample from bundle
│   └── job_description.docx       # Original JD from bundle
│
└── output/
    └── submission.csv             # Ranked top-100 output
```

---

## Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0000031,1,1.0000,"Recommendation Systems Engineer, 6.0yr, Hyderabad, India. Retrieval skills: Faiss, Pinecone, Embeddings, Sentence Transformers. Product cos: Swiggy, Mad Street Den. Responsive (91% recruiter reply rate). 60d notice."
CAND_0000038,2,0.4532,"..."
...
```

- `score` — normalised 0.0–1.0, strictly non-increasing
- `reasoning` — specific to each candidate, no templating

---

## Sample Results (50-candidate dataset)

| Rank | Candidate ID | Title | Score | Why |
|---|---|---|---|---|
| #1 | CAND_0000031 | Recommendation Systems Engineer | 74.4 | FAISS + Pinecone + Embeddings · Swiggy/Zomato/Uber · 91% response rate |
| #2 | CAND_0000038 | Java Developer | 38.9 | Weaviate + MLOps · Swiggy · Open to work |
| #3 | CAND_0000043 | Cloud Engineer | 43.8 | Elasticsearch + Haystack + PEFT · Low availability (4% response rate) |

**Distribution from 50-candidate sample:**
- 9 potential fits
- 41 disqualified (consulting-only or wrong domain)
- 0 honeypots detected

---

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
  ┌─────────────┐
  │  Honeypot   │  ← Timeline check, expert-skill-zero-duration check
  │  Detection  │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  JD-based   │  ← Consulting-only, wrong domain, no ML evidence
  │ Disqualifier│
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  6-Dimension│  ← Weighted composite score (0–100)
  │   Scorer    │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Behavioral │  ← 23 Redrob signals → multiplier (0.10–1.0)
  │  Multiplier │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Sort + Top │  ← Non-increasing scores, candidate_id tiebreak
  │     100     │
  └─────────────┘
```

---

## AI Tools Declaration

- Claude (Anthropic) — architecture discussion, code review
- No candidate data was fed to any external LLM
- Ranking is fully deterministic — no LLM calls during ranking

---

*Built for the Redrob Intelligent Candidate Discovery & Ranking Hackathon.*
