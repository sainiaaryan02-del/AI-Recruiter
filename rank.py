#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranking Engine
============================================
Job: Senior AI Engineer — Founding Team, Redrob AI, Pune/Noida

Architecture:
  Score = Composite(6 dimensions) × Behavioral_Multiplier
  
Composite weights (tuned to actual JD requirements):
  1. Retrieval & Ranking Skills  — 30%
  2. Career Context              — 26%
  3. Applied ML Depth            — 18%
  4. Experience Band             — 12%
  5. Location & Logistics        — 9%
  6. Education                   — 5%

Behavioral multiplier (from 23 Redrob signals): 0.10–1.0
Final score = composite × multiplier, normalised to 0–1
"""

import json, re, math, csv, sys, os
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

# ── Reference date ────────────────────────────────────────────────────────────
TODAY = date(2026, 6, 27)

# ── JD-derived vocabularies ───────────────────────────────────────────────────
# "Things you absolutely need" — direct from JD
MUST_HAVE_RETRIEVAL = {
    "embeddings", "sentence transformers", "sentence-transformers",
    "openai embeddings", "bge", "e5", "text embeddings",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "information retrieval",
    "hybrid search", "vector search", "vector database", "semantic search",
    "dense retrieval",
}

# "Things we'd like" — preferred skills from JD
PREFERRED_SKILLS = {
    "fine-tuning llms", "lora", "qlora", "peft",
    "learning to rank", "xgboost", "lightgbm", "neural ranking",
    "mlflow", "mlops", "feature engineering",
    "pytorch", "tensorflow", "hugging face transformers",
    "langchain", "haystack", "bm25", "nlp", "transformers",
    "machine learning", "deep learning", "recommendation systems",
    "bentoml", "weights & biases", "scikit-learn",
    "kubeflow", "distributed systems",
}

# "Things we explicitly do NOT want" — direct from JD
CONSULTING_ONLY = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hexaware", "hcl",
    "mphasis", "mindtree", "l&t infotech",
}

# Product companies — positive signals
PRODUCT_COS = {
    "swiggy", "zomato", "flipkart", "ola", "uber", "cred", "razorpay",
    "phonepe", "paytm", "meesho", "myntra", "nykaa", "dream11",
    "mad street den", "pied piper", "hooli", "wayne enterprises",
    "stark industries", "google", "microsoft", "amazon", "meta",
    "netflix", "airbnb", "nvidia", "openai", "anthropic",
}

# Wrong domains from JD: CV, speech, robotics without NLP/IR
WRONG_DOMAIN_SKILLS = {
    "opencv", "yolo", "object detection", "image classification",
    "computer vision", "speech recognition", "tts", "gans", "cnn",
}

# Clearly non-ML job titles (trap candidates)
WRONG_TITLES = {
    "marketing manager", "hr manager", "operations manager",
    "accountant", "civil engineer", "mechanical engineer",
    "graphic designer", "customer support", "business analyst",
    "project manager", "frontend engineer", "mobile developer",
    ".net developer",
}

# India preferred locations from JD
PREFERRED_LOCS = {
    "pune", "noida", "hyderabad", "bangalore", "bengaluru",
    "mumbai", "delhi", "gurgaon", "gurugram", "chandigarh",
    "trivandrum", "kochi", "india",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def days_since(date_str: str) -> int:
    try:
        return (TODAY - datetime.strptime(date_str, "%Y-%m-%d").date()).days
    except Exception:
        return 999


def get_skills(candidate: Dict) -> Tuple[set, Dict, Dict]:
    """Returns (lowercase name set, proficiency dict, duration dict)."""
    names, prof, dur = set(), {}, {}
    for s in candidate.get("skills", []):
        n = s["name"].lower().strip()
        names.add(n)
        prof[n] = s.get("proficiency", "beginner")
        dur[n]  = s.get("duration_months", 0)
    return names, prof, dur


def career_text(candidate: Dict) -> str:
    parts = []
    for ch in candidate.get("career_history", []):
        parts += [ch.get("title",""), ch.get("description",""),
                  ch.get("industry",""), ch.get("company","")]
    return " ".join(parts).lower()


def companies(candidate: Dict) -> List[str]:
    return [ch.get("company","").lower() for ch in candidate.get("career_history",[])]


def is_consulting_only(candidate: Dict) -> bool:
    cos = companies(candidate)
    if not cos:
        return False
    return all(any(cf in co for cf in CONSULTING_ONLY) for co in cos)


def honeypot_check(candidate: Dict) -> Tuple[bool, str]:
    """Return (is_honeypot, reason) based on profile consistency."""
    p = candidate["profile"]
    yoe_mo = p["years_of_experience"] * 12
    career_mo = sum(ch.get("duration_months", 0)
                    for ch in candidate.get("career_history", []))

    # Timeline impossibility: more than 3 years gap
    if career_mo > 0 and (yoe_mo - career_mo) > 36:
        return True, f"YoE {p['years_of_experience']}yr claimed but career = {career_mo/12:.1f}yr"

    # Expert/advanced skills with zero months used (≥4 such skills)
    zero_expert = [s for s in candidate.get("skills", [])
                   if s.get("proficiency") in ("expert","advanced")
                   and s.get("duration_months", 0) == 0]
    if len(zero_expert) >= 4:
        return True, f"{len(zero_expert)} expert/advanced skills with 0 months used"

    # Salary min > max
    sal = candidate.get("redrob_signals",{}).get("expected_salary_range_inr_lpa",{})
    if sal.get("min",0) > sal.get("max",0) > 0:
        return True, "Salary min > max"

    return False, ""


# ── Scoring dataclasses ────────────────────────────────────────────────────────

@dataclass
class Dims:
    retrieval:    float = 0.0   # 30%
    career:       float = 0.0   # 26%
    ml_depth:     float = 0.0   # 18%
    exp_band:     float = 0.0   # 12%
    logistics:    float = 0.0   # 9%
    education:    float = 0.0   # 5%

    def composite(self) -> float:
        return (0.30 * self.retrieval
              + 0.26 * self.career
              + 0.18 * self.ml_depth
              + 0.12 * self.exp_band
              + 0.09 * self.logistics
              + 0.05 * self.education)


@dataclass
class Behavioral:
    availability: float = 0.0   # 40% of multiplier
    engagement:   float = 0.0   # 38%
    credibility:  float = 0.0   # 22%

    def multiplier(self) -> float:
        raw = 0.40*self.availability + 0.38*self.engagement + 0.22*self.credibility
        return max(0.10, min(1.0, raw))


@dataclass
class Result:
    rank:          int
    candidate_id:  str
    name:          str
    title:         str
    yoe:           float
    location:      str
    country:       str
    composite:     float
    bm:            float
    final:         float
    dims:          Dims
    beh:           Behavioral
    is_honeypot:   bool
    disqualified:  bool
    dq_reason:     str
    reasoning:     str = ""


# ── Dimension scorers ─────────────────────────────────────────────────────────

def score_retrieval(candidate: Dict) -> float:
    """
    Dim 1 (30%): Production retrieval/ranking experience.
    Directly maps to JD 'Things you absolutely need'.
    """
    skills, prof, dur = get_skills(candidate)
    text = career_text(candidate)
    pw = {"expert":1.0, "advanced":0.75, "intermediate":0.5, "beginner":0.25}
    score = 0.0

    # A) Explicit retrieval infrastructure skills (highest weight)
    ret_hits = [(s, pw.get(prof.get(s,"beginner"),0.25), dur.get(s,0))
                for s in MUST_HAVE_RETRIEVAL if s in skills]
    if ret_hits:
        # Sum of proficiency × log(duration+1), normalised
        raw = sum(p * math.log1p(d) / math.log1p(60) for _, p, d in ret_hits)
        score += min(0.45, raw * 0.20)
        # Bonus for multiple distinct retrieval tools (operational breadth)
        score += min(0.10, len(ret_hits) * 0.03)

    # B) Evaluation framework knowledge (NDCG, MRR, MAP, A/B)
    eval_kw = ["ndcg","mrr","map","a/b test","offline eval","ranking metric",
               "retrieval quality","relevance","precision@","recall@"]
    eval_hits = sum(1 for kw in eval_kw if kw in text)
    score += min(0.20, eval_hits * 0.07)

    # C) Evidence of shipping (from career descriptions)
    ship_kw = ["shipped","deployed","production","launched","real users",
               "at scale","serving","embedding-based","ranking model",
               "search product","discovery feed","retrieval system",
               "recommendation engine","hybrid retrieval","vector search"]
    ship_hits = sum(1 for kw in ship_kw if kw in text)
    score += min(0.25, ship_hits * 0.025)

    return min(1.0, score)


def score_career(candidate: Dict) -> float:
    """
    Dim 2 (26%): Career context — product cos, title fit, trajectory.
    JD explicitly disqualifies: consulting-only, title-chasers, wrong domain.
    """
    p    = candidate["profile"]
    hist = candidate.get("career_history", [])
    cos  = companies(candidate)
    text = career_text(candidate)
    title = p["current_title"].lower()
    score = 0.0

    # A) Current title sanity check
    is_wrong_title = any(wt in title for wt in WRONG_TITLES)
    has_ml_career  = any(kw in text for kw in
        ["machine learning","retrieval","ranking","embedding","nlp","deep learning",
         "recommendation","ai engineer","ml engineer","search","vector"])

    if is_wrong_title and not has_ml_career:
        return 0.05   # Strong disqualifier signal — near-zero score

    # B) Product company experience
    product_stints  = sum(1 for co in cos if any(pc in co for pc in PRODUCT_COS))
    consult_stints  = sum(1 for co in cos if any(cf in co for cf in CONSULTING_ONLY))

    if product_stints >= 3:
        score += 0.35
    elif product_stints == 2:
        score += 0.28
    elif product_stints == 1:
        score += 0.15
    # consulting-only already handled by disqualifier; partial penalty here
    if consult_stints > 0 and product_stints == 0:
        score -= 0.08

    # C) Industry relevance — time in tech / AI / product industries
    ai_industries = {"ai/ml","food delivery","transportation","fintech",
                     "e-commerce","software","technology","internet","saas"}
    for ch in hist:
        ind = ch.get("industry","").lower()
        if any(ai_ind in ind for ai_ind in ai_industries):
            score += min(0.06, ch.get("duration_months",0) / 12 * 0.015)

    # D) Anti title-chaser: penalise many short stints
    short_stints = sum(1 for ch in hist if ch.get("duration_months",0) < 18)
    if short_stints >= 3:
        score -= 0.10

    # E) Leadership / ownership language (JD wants founding-team mindset)
    lead_kw = ["led","owned","drove","architected","built and shipped",
               "built the","mentored","founded","designed and implemented"]
    lead_hits = sum(1 for kw in lead_kw if kw in text)
    score += min(0.15, lead_hits * 0.04)

    # F) Wrong-domain penalty (CV/speech/robotics without IR — from JD)
    skills, _, dur = get_skills(candidate)
    wrong_d_heavy = [s for s in WRONG_DOMAIN_SKILLS
                     if s in skills and dur.get(s,0) >= 24]
    has_retrieval = bool(MUST_HAVE_RETRIEVAL & skills)
    if len(wrong_d_heavy) >= 3 and not has_retrieval:
        score -= 0.12

    return max(0.0, min(1.0, score))


def score_ml_depth(candidate: Dict) -> float:
    """
    Dim 3 (18%): Actual ML depth — pre-LLM era experience matters.
    JD: 'people who understood retrieval before it became fashionable'.
    """
    skills, prof, dur = get_skills(candidate)
    text  = career_text(candidate)
    pw    = {"expert":1.0,"advanced":0.75,"intermediate":0.5,"beginner":0.25}
    score = 0.0

    # A) Pre-LLM era ML (JD explicitly values this)
    pre_llm = ["xgboost","lightgbm","gradient boosting","scikit-learn",
               "feature engineering","learning to rank","bm25","inverted index",
               "collaborative filtering","matrix factorization","ranking model",
               "catboost","random forest","sklearn"]
    pre_llm_hits = sum(1 for kw in pre_llm if kw in text or kw in skills)
    score += min(0.25, pre_llm_hits * 0.06)

    # B) Modern ML stack (weighted by proficiency × depth)
    modern = {
        "pytorch":0.10, "tensorflow":0.07,
        "hugging face transformers":0.10, "sentence transformers":0.12,
        "fine-tuning llms":0.10, "peft":0.08, "lora":0.08,
        "mlops":0.07, "mlflow":0.06, "weights & biases":0.06,
        "deep learning":0.08, "machine learning":0.06,
        "langchain":0.04, "haystack":0.08,
        "embeddings":0.10, "recommendation systems":0.08,
        "bentoml":0.05, "kubeflow":0.06,
    }
    for sk, val in modern.items():
        if sk in skills:
            p_w = pw.get(prof.get(sk,"beginner"), 0.25)
            score += val * p_w

    # C) Skill depth: expert/advanced with ≥24 months
    deep = [s for s in skills
            if prof.get(s) in ("expert","advanced")
            and dur.get(s,0) >= 24
            and s in (MUST_HAVE_RETRIEVAL | PREFERRED_SKILLS)]
    score += min(0.20, len(deep) * 0.06)

    # D) Platform assessment scores (verified competency)
    asses = candidate.get("redrob_signals",{}).get("skill_assessment_scores",{})
    rel_asses = [v for k, v in asses.items()
                 if k.lower() in (MUST_HAVE_RETRIEVAL | PREFERRED_SKILLS)]
    if rel_asses:
        score += (sum(rel_asses) / len(rel_asses) / 100) * 0.15

    return max(0.0, min(1.0, score))


def score_exp_band(candidate: Dict) -> float:
    """
    Dim 4 (12%): Experience band fit + disqualifier patterns.
    JD: '5-9 years' band, but judgment over years.
    """
    p    = candidate["profile"]
    yoe  = p["years_of_experience"]
    text = career_text(candidate)

    if yoe < 1.5:
        return 0.0

    # Pure academic / research-only disqualifier (from JD)
    is_research = (any(kw in text for kw in ["research lab","phd","academic paper"])
                   and not any(kw in text for kw in
                               ["deployed","production","shipped","real users","at scale"]))
    if is_research:
        return 0.10

    # YoE band
    if 5 <= yoe <= 9:
        band = 1.0
    elif 4 <= yoe < 5:
        band = 0.80
    elif 9 < yoe <= 11:
        band = 0.72
    elif 3 <= yoe < 4:
        band = 0.55
    elif yoe > 11:
        band = 0.55
    else:
        band = 0.30

    # JD disqualifier: senior eng not writing code in 18+ months
    arch_only = any(kw in text for kw in
        ["architecture only","no longer codes","moved to management",
         "purely strategic","not writing code"])
    if arch_only:
        band *= 0.6

    return min(1.0, band)


def score_logistics(candidate: Dict) -> float:
    """
    Dim 5 (9%): Location, notice period, work mode.
    JD: Pune/Noida preferred; sub-30d notice ideal; case-by-case outside India.
    """
    p   = candidate["profile"]
    rs  = candidate.get("redrob_signals",{})
    loc = p.get("location","").lower()
    cty = p.get("country","").lower()
    notice   = rs.get("notice_period_days", 90)
    relocate = rs.get("willing_to_relocate", False)
    mode     = rs.get("preferred_work_mode","flexible")
    score    = 0.0

    # Location
    in_india    = "india" in cty
    in_preferred = any(city in loc for city in PREFERRED_LOCS)

    if in_preferred and in_india:
        score += 0.52
    elif in_india:
        score += 0.30
        if relocate:
            score += 0.18
    elif relocate:
        score += 0.18   # outside India but willing to move

    # Notice period
    if notice <= 30:
        score += 0.35
    elif notice <= 60:
        score += 0.22
    elif notice <= 90:
        score += 0.10
    # 90+ = 0 bonus (difficult per JD)

    # Work mode compatibility (hybrid/onsite preferred for founding team)
    if mode in ("hybrid","flexible","onsite"):
        score += 0.13

    return min(1.0, score)


def score_education(candidate: Dict) -> float:
    """Dim 6 (5%): Tier × field relevance."""
    edu = candidate.get("education", [])
    if not edu:
        return 0.25
    best = 0.0
    for e in edu:
        tier  = {"tier_1":1.0,"tier_2":0.75,"tier_3":0.50,
                 "tier_4":0.28,"unknown":0.22}.get(e.get("tier","unknown"),0.22)
        field = e.get("field_of_study","").lower()
        fld   = 0.0
        if any(f in field for f in ["computer","software","information","data","ai",
                                     "machine learning","artificial"]):
            fld = 0.30
        elif any(f in field for f in ["electronics","electrical","math","statistics","physics"]):
            fld = 0.15
        best = max(best, 0.70*tier + 0.30*fld)
    return min(1.0, best)


def score_behavioral(candidate: Dict) -> Behavioral:
    """
    Convert 23 Redrob signals into an availability multiplier (0.10–1.0).
    A great-on-paper candidate who is unreachable must be down-weighted.
    """
    rs  = candidate.get("redrob_signals", {})
    beh = Behavioral()

    # ── Availability (40%) ────────────────────────────────────────────────────
    av = 0.0
    if rs.get("open_to_work_flag", False):
        av += 0.32
    # Recency: exponential decay, half-life 14 days
    last_active = days_since(rs.get("last_active_date","2020-01-01"))
    av += 0.33 * math.exp(-last_active / 14)
    # Active job-seeking
    apps = rs.get("applications_submitted_30d", 0)
    av += min(0.18, apps * 0.045)
    # Notice period
    notice = rs.get("notice_period_days", 90)
    if notice <= 30:
        av += 0.17
    elif notice <= 60:
        av += 0.09
    beh.availability = min(1.0, av)

    # ── Engagement quality (38%) ──────────────────────────────────────────────
    eng = 0.0
    resp = rs.get("recruiter_response_rate", 0.0)
    eng += 0.36 * resp
    # Response time
    rtime = rs.get("avg_response_time_hours", 999)
    if rtime <= 24:
        eng += 0.22
    elif rtime <= 72:
        eng += 0.13
    elif rtime <= 168:
        eng += 0.05
    # Interview attendance
    icr = rs.get("interview_completion_rate", 0.5)
    eng += 0.24 * icr
    # Offer acceptance (historical reliability)
    oar = rs.get("offer_acceptance_rate", -1)
    if oar >= 0:
        eng += 0.18 * oar
    beh.engagement = min(1.0, eng)

    # ── Profile credibility (22%) ─────────────────────────────────────────────
    cred = 0.0
    cred += rs.get("profile_completeness_score", 50) / 100 * 0.35
    if rs.get("verified_email", False):
        cred += 0.15
    if rs.get("verified_phone", False):
        cred += 0.10
    if rs.get("linkedin_connected", False):
        cred += 0.10
    github = rs.get("github_activity_score", -1)
    if github >= 0:
        cred += (github / 100) * 0.20
    saved = rs.get("saved_by_recruiters_30d", 0)
    cred += min(0.10, saved * 0.02)
    beh.credibility = min(1.0, cred)

    return beh


# ── Reasoning builder ─────────────────────────────────────────────────────────

def build_reasoning(c: Dict, dims: Dims, beh: Behavioral,
                    is_hp: bool, disq: bool, dq_reason: str) -> str:
    if is_hp:
        return f"Flagged honeypot — {dq_reason}."
    if disq:
        p = c["profile"]
        return (f"Disqualified: {dq_reason}. "
                f"Title: {p['current_title']}, {p['years_of_experience']}yrs.")

    p     = c["profile"]
    rs    = c.get("redrob_signals", {})
    skills, prof, dur = get_skills(c)
    cos   = [ch.get("company","") for ch in c.get("career_history",[])]
    text  = career_text(c)

    parts = []

    # Identity
    loc = f"{p.get('location','')}, {p.get('country','')}".strip(", ")
    parts.append(f"{p['current_title']}, {p['years_of_experience']}yr, {loc}.")

    # Top retrieval skills present
    ret_found = [s.title() for s in sorted(MUST_HAVE_RETRIEVAL)
                 if s in skills][:4]
    if ret_found:
        parts.append(f"Retrieval skills: {', '.join(ret_found)}.")

    # Product company experience
    prod_cos = [co for co in cos if any(pc in co.lower() for pc in PRODUCT_COS)]
    if prod_cos:
        parts.append(f"Product cos: {', '.join(prod_cos[:2])}.")

    # Availability signals
    notice    = rs.get("notice_period_days", 90)
    resp      = rs.get("recruiter_response_rate", 0)
    last_days = days_since(rs.get("last_active_date","2020-01-01"))
    github    = rs.get("github_activity_score", -1)

    if resp >= 0.7:
        parts.append(f"Responsive ({resp:.0%} recruiter reply rate).")
    elif resp < 0.20:
        parts.append(f"Low response rate ({resp:.0%}) — may be hard to reach.")

    if last_days <= 7:
        parts.append(f"Active {last_days}d ago.")
    elif last_days > 90:
        parts.append(f"Inactive {last_days}d — availability uncertain.")

    if notice <= 30:
        parts.append(f"{notice}d notice.")
    elif notice >= 90:
        parts.append(f"Long notice ({notice}d).")

    if github >= 40:
        parts.append(f"GitHub activity score {github:.0f}.")

    return " ".join(parts)[:500]


# ── Main ranking function ─────────────────────────────────────────────────────

def rank_candidates(candidates: List[Dict]) -> List[Result]:
    results = []

    for c in candidates:
        p = c["profile"]

        # 1) Honeypot detection
        is_hp, hp_reason = honeypot_check(c)

        # 2) Hard disqualifier checks (from JD)
        disq, dq_reason = False, ""

        if is_consulting_only(c):
            disq, dq_reason = True, "Consulting-only career (TCS/Infosys/Wipro/etc)"

        title_lc = p["current_title"].lower()
        text     = career_text(c)
        if any(wt in title_lc for wt in WRONG_TITLES):
            has_ml = any(kw in text for kw in
                ["machine learning","retrieval","ranking","embedding","recommendation",
                 "nlp","deep learning","ai engineer","ml engineer","vector search"])
            if not has_ml:
                disq     = True
                dq_reason = f"Wrong domain title '{p['current_title']}' with no ML evidence"

        # 3) Score all dimensions
        dims = Dims(
            retrieval  = score_retrieval(c),
            career     = score_career(c),
            ml_depth   = score_ml_depth(c),
            exp_band   = score_exp_band(c),
            logistics  = score_logistics(c),
            education  = score_education(c),
        )
        beh = score_behavioral(c)

        comp = dims.composite() * 100   # 0–100

        # Apply penalties
        if is_hp:
            comp = 0.0
        elif disq:
            comp = min(comp, 12.0) * (0.9 + 0.1 * beh.multiplier())

        final = (comp / 100) * beh.multiplier()

        reasoning = build_reasoning(c, dims, beh, is_hp, disq, dq_reason)

        results.append(Result(
            rank         = 0,
            candidate_id = c["candidate_id"],
            name         = p.get("anonymized_name",""),
            title        = p["current_title"],
            yoe          = p["years_of_experience"],
            location     = p.get("location",""),
            country      = p.get("country",""),
            composite    = round(comp, 2),
            bm           = round(beh.multiplier(), 3),
            final        = round(final, 6),
            dims         = dims,
            beh          = beh,
            is_honeypot  = is_hp,
            disqualified = disq,
            dq_reason    = dq_reason if disq else (hp_reason if is_hp else ""),
            reasoning    = reasoning,
        ))

    # Sort: good candidates first (by final desc), then DQ, then honeypots
    def sort_key(r):
        tier = 2 if r.is_honeypot else (1 if r.disqualified else 0)
        return (tier, -r.final, r.candidate_id)   # candidate_id ascending for tie-break

    results.sort(key=sort_key)
    for i, r in enumerate(results, 1):
        r.rank = i
    return results


# ── CSV writer ────────────────────────────────────────────────────────────────

def write_csv(ranked: List[Result], path: str, n: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    top = ranked[:n]

    # Build scores that are strictly non-increasing with unique 4-dp values
    raw = [r.final for r in top]
    max_s = raw[0] if raw else 1.0
    if max_s <= 0:
        max_s = 1.0

    # Spread linearly within each "equal block" to ensure unique 4-dp values
    scores_out = []
    prev = None
    for idx, s in enumerate(raw):
        norm = round(s / max_s, 6)
        # Ensure strictly decreasing at 4dp so equal scores get candidate_id tiebreak
        # We use the row index as a tiny epsilon (1e-6 per rank)
        val = round(max(0.0, norm - idx * 1e-6), 4)
        if prev is not None and val >= prev:
            val = round(prev - 0.0001, 4)
        scores_out.append(max(0.0, val))
        prev = scores_out[-1]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id","rank","score","reasoning"])
        w.writeheader()
        for i, (r, s) in enumerate(zip(top, scores_out), 1):
            w.writerow({
                "candidate_id": r.candidate_id,
                "rank":         str(i),
                "score":        f"{s:.4f}",
                "reasoning":    r.reasoning,
            })
    print(f"✅  Written → {path}  ({len(top)} rows)")


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(ranked: List[Result]) -> None:
    print("\n" + "═"*108)
    print("  REDROB — Senior AI Engineer Ranking")
    print("═"*108)
    print(f"\n{'Rk':>3}  {'Score':>7}  {'Comp':>5}  {'BM':>5}  {'ID':<16}  {'Title':<36}  {'YoE':>4}  Loc")
    print("─"*108)
    for r in ranked:
        flag = " [HP]" if r.is_honeypot else (" [DQ]" if r.disqualified else "")
        print(f"{r.rank:>3}  {r.final:>7.5f}  {r.composite:>5.1f}  {r.bm:>5.3f}  "
              f"{r.candidate_id:<16}  {(r.title+flag):<36}  {r.yoe:>4.1f}  "
              f"{r.location[:22]},{r.country[:5]}")

    print(f"\n{'═'*108}")
    print(f"\n── Dimension breakdown — Top 15 ──────────────────────────────────────────────────")
    print(f"{'ID':<16}  {'Retr':>5}  {'Car':>5}  {'ML':>5}  {'Exp':>5}  {'Log':>5}  {'Edu':>5}  {'BM':>6}  {'Final':>7}")
    print("─"*80)
    for r in ranked[:15]:
        d = r.dims
        print(f"{r.candidate_id:<16}  {d.retrieval:>5.2f}  {d.career:>5.2f}  "
              f"{d.ml_depth:>5.2f}  {d.exp_band:>5.2f}  {d.logistics:>5.2f}  "
              f"{d.education:>5.2f}  {r.bm:>6.3f}  {r.final:>7.5f}")

    print(f"\n── Reasoning — Top 10 ───────────────────────────────────────────────────────────")
    for r in ranked[:10]:
        print(f"\n#{r.rank} {r.candidate_id} | {r.title} | {r.yoe}yr | {r.location}")
        print(f"   {r.reasoning}")

    hp  = sum(1 for r in ranked if r.is_honeypot)
    dq  = sum(1 for r in ranked if r.disqualified and not r.is_honeypot)
    ok  = len(ranked) - hp - dq
    print(f"\n── Summary ────")
    print(f"  Total: {len(ranked)}   Good: {ok}   Disqualified: {dq}   Honeypots: {hp}")
    print("═"*108 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default="/mnt/user-data/uploads/sample_candidates.json")
    ap.add_argument("--out",  default="/home/claude/redrob/output/submission.csv")
    ap.add_argument("--top",  type=int, default=50)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.candidates} ...")
    with open(args.candidates) as f:
        raw = f.read().strip()
    candidates = json.loads(raw) if raw.startswith("[") else \
                 [json.loads(l) for l in raw.splitlines() if l.strip()]
    print(f"Loaded {len(candidates)} candidates.")

    ranked = rank_candidates(candidates)

    if not args.quiet:
        print_report(ranked)

    write_csv(ranked, args.out, n=args.top)


if __name__ == "__main__":
    main()
