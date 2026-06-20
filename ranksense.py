"""
RankSense — single-file edition
================================
An explainable AI engine for JD-to-Candidate matching & ranking.

This file combines everything from the multi-module version (config,
parsing, embeddings, scoring, validation, explainability, ranking, the
FastAPI API, and the dashboard frontend) into ONE file for easy copy-paste
into a single GitHub repo / VS Code file.

Run it:
    pip install fastapi uvicorn python-multipart scikit-learn numpy pypdf python-docx python-dotenv
    python ranksense.py
    # then open http://localhost:8000

Or with uvicorn directly:
    uvicorn ranksense:app --reload --port 8000

Optional (better semantic matching, downloads a small model on first run):
    pip install sentence-transformers

Optional (LLM-enhanced JD parsing — set this env var to enable):
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install anthropic

Run tests (uses the offline TF-IDF backend automatically):
    pip install pytest httpx
    pytest ranksense.py

Run the CLI demo (ranks the built-in sample resumes against a sample JD,
no server needed):
    python ranksense.py --demo
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════
class Settings:
    # --- Scoring weights (must sum to 1.0) ---
    WEIGHT_SKILL_MATCH: float = 0.50
    WEIGHT_EXPERIENCE: float = 0.25
    WEIGHT_SEMANTIC: float = 0.25

    # --- Thresholds ---
    SKILL_MATCH_THRESHOLD: float = 0.55
    MIN_RESUME_WORDS: int = 40
    WEAK_MATCH_THRESHOLD: float = 40.0
    EXPERIENCE_OVERQUALIFY_BUFFER_YEARS: float = 2.0

    # --- Embeddings: "auto" | "sentence-transformers" | "tfidf" ---
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "auto")
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

    # --- Optional LLM-enhanced JD parsing ---
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # --- Skill vocabulary used by the rule-based extractor/parser ---
    SKILL_VOCABULARY: list[str] = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
        "spring boot", "machine learning", "deep learning", "nlp",
        "data analysis", "data engineering", "etl", "spark", "hadoop",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
        "rest api", "graphql", "microservices", "ci/cd", "git", "linux",
        "agile", "scrum", "project management", "product management",
        "communication", "leadership", "stakeholder management",
        "sales", "negotiation", "crm", "salesforce", "seo", "sem",
        "figma", "ui/ux design", "excel", "power bi", "tableau",
    ]

    def validate(self) -> None:
        total = self.WEIGHT_SKILL_MATCH + self.WEIGHT_EXPERIENCE + self.WEIGHT_SEMANTIC
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}.")


settings = Settings()
settings.validate()


# ════════════════════════════════════════════════════════════════════════
#  MODELS (Pydantic request/response schemas)
# ════════════════════════════════════════════════════════════════════════
class JDParseRequest(BaseModel):
    jd_text: str = Field(..., min_length=20, description="Raw job description text")


class ParsedJD(BaseModel):
    title: str | None = None
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    min_experience_years: float | None
    seniority: str | None
    raw_text: str


class SkillEvidence(BaseModel):
    skill: str
    matched: bool
    similarity: float
    evidence_sentence: str | None = None


class ScoreBreakdown(BaseModel):
    skill_match_score: float
    experience_score: float
    semantic_score: float
    final_score: float


class DataQualityFlag(BaseModel):
    is_sparse: bool = False
    is_duplicate: bool = False
    word_count: int = 0
    notes: list[str] = []


class CandidateResult(BaseModel):
    candidate_id: str
    filename: str
    candidate_name: str | None
    detected_experience_years: float | None
    score: ScoreBreakdown
    must_have_evidence: list[SkillEvidence]
    nice_to_have_evidence: list[SkillEvidence]
    missing_must_haves: list[str]
    data_quality: DataQualityFlag
    summary: str


class RankResponse(BaseModel):
    parsed_jd: ParsedJD
    candidates: list[CandidateResult]
    embedding_backend: str


# ════════════════════════════════════════════════════════════════════════
#  JD PARSER  —  JD Understanding
# ════════════════════════════════════════════════════════════════════════
SENIORITY_PATTERNS = [
    (r"\b(intern|internship)\b", "Intern"),
    (r"\b(junior|jr\.?)\b", "Junior"),
    (r"\b(senior|sr\.?)\b", "Senior"),
    (r"\b(lead|principal|staff)\b", "Lead/Principal"),
    (r"\b(manager|head of|director)\b", "Manager/Director"),
    (r"\b(mid[- ]level|intermediate)\b", "Mid-level"),
]

NICE_TO_HAVE_MARKERS = [
    "nice to have", "good to have", "preferred", "bonus", "plus",
    "is a plus", "added advantage", "would be a plus", "optional",
]

EXPERIENCE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-|to)?\s*\d*\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_skills(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for skill in settings.SKILL_VOCABULARY:
        variant = re.escape(skill).replace(r"\.", r"[.\s]?").replace(r"\ ", r"[\s/-]?")
        pattern = rf"\b{variant}\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


NICE_TO_HAVE_SECTION_HEADERS = [
    "nice to have", "good to have", "preferred qualifications", "preferred skills",
    "bonus points", "added advantage", "optional",
]
MUST_HAVE_SECTION_HEADERS = [
    "requirements", "must have", "must-have", "required qualifications",
    "responsibilities", "qualifications", "what you'll need",
    "minimum qualifications", "key responsibilities",
]


def _is_section_header(line_lower: str, headers: list[str]) -> bool:
    return any(line_lower == h or line_lower.startswith(h) for h in headers)


def _split_must_vs_nice(original_text: str, all_skills: list[str]) -> tuple[list[str], list[str]]:
    """Section-aware split: walks the JD line-by-line (preserving structure),
    tracking whether we're currently inside a 'Nice to have' style section
    (vs. 'Requirements'), plus catching inline markers like
    'GraphQL is a plus' on a single line regardless of section."""
    nice_to_have: set[str] = set()
    in_nice_section = False

    for line in original_text.splitlines():
        raw_line = line.strip()
        if not raw_line:
            continue
        is_bullet = raw_line.startswith(("-", "•", "‣", "·", "*"))
        stripped = raw_line.lstrip("-•‣·* ").strip()
        lower = stripped.lower().rstrip(":")

        if len(stripped) < 60:
            if _is_section_header(lower, NICE_TO_HAVE_SECTION_HEADERS):
                in_nice_section = True
                continue
            if _is_section_header(lower, MUST_HAVE_SECTION_HEADERS):
                in_nice_section = False
                continue

        inline_marker = any(marker in lower for marker in NICE_TO_HAVE_MARKERS)

        # A non-bullet, non-header sentence signals we've left the bullet
        # list (e.g. a closing paragraph after a "Nice to have" list) —
        # exit the section unless this line itself carries an inline marker.
        if not is_bullet and not inline_marker:
            in_nice_section = False

        if in_nice_section or inline_marker:
            for skill in all_skills:
                variant = re.escape(skill).replace(r"\.", r"[.\s]?").replace(r"\ ", r"[\s/-]?")
                if re.search(rf"\b{variant}\b", lower):
                    nice_to_have.add(skill)

    must_have = [s for s in all_skills if s not in nice_to_have]
    return must_have, sorted(nice_to_have)


def _extract_min_experience(text: str) -> float | None:
    matches = EXPERIENCE_PATTERN.findall(text)
    if not matches:
        return None
    years = [float(m) for m in matches]
    return min(years)


def _extract_seniority(text: str) -> str | None:
    text_lower = text.lower()
    for pattern, label in SENIORITY_PATTERNS:
        if re.search(pattern, text_lower):
            return label
    return None


def _extract_title(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if 3 < len(line) <= 80 and not line.endswith(":"):
            return line
    return None


def parse_jd_rule_based(jd_text: str) -> ParsedJD:
    normalized = _normalize(jd_text)
    all_skills = _find_skills(normalized)
    must_have, nice_to_have = _split_must_vs_nice(jd_text, all_skills)

    return ParsedJD(
        title=_extract_title(jd_text),
        must_have_skills=must_have,
        nice_to_have_skills=nice_to_have,
        min_experience_years=_extract_min_experience(normalized),
        seniority=_extract_seniority(normalized),
        raw_text=normalized,
    )


def _enhance_with_llm(parsed: ParsedJD, jd_text: str) -> ParsedJD:
    """Optional: refine extraction with Claude. No-ops without an API key,
    or if the call fails for any reason — rule-based result always stands."""
    if not settings.ANTHROPIC_API_KEY:
        return parsed
    try:
        import json
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        prompt = (
            "Extract hiring requirements from this job description as strict JSON "
            'with keys "must_have_skills" (array of strings) and '
            '"nice_to_have_skills" (array of strings). Only include skills not '
            f"already in this list: {parsed.must_have_skills + parsed.nice_to_have_skills}. "
            "Return ONLY the JSON object, nothing else.\n\n"
            f"Job description:\n{jd_text[:4000]}"
        )
        resp = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(block.text for block in resp.content if hasattr(block, "text"))
        raw = raw.strip().strip("```json").strip("```").strip()
        extra = json.loads(raw)
        parsed.must_have_skills = sorted(
            set(parsed.must_have_skills) | set(extra.get("must_have_skills", []))
        )
        parsed.nice_to_have_skills = sorted(
            set(parsed.nice_to_have_skills) | set(extra.get("nice_to_have_skills", []))
        )
    except Exception:
        pass
    return parsed


def parse_jd(jd_text: str) -> ParsedJD:
    parsed = parse_jd_rule_based(jd_text)
    parsed = _enhance_with_llm(parsed, jd_text)
    return parsed


# ════════════════════════════════════════════════════════════════════════
#  RESUME PARSER
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ParsedResume:
    filename: str
    raw_text: str
    sentences: list[str]
    candidate_name: str | None
    detected_skills: list[str] = field(default_factory=list)
    detected_experience_years: float | None = None
    word_count: int = 0


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_text_from_docx(file_bytes: bytes) -> str:
    import docx
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs)


def _extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_text_from_pdf(file_bytes)
    if lower.endswith(".docx"):
        return _extract_text_from_docx(file_bytes)
    if lower.endswith(".txt"):
        return _extract_text_from_txt(file_bytes)
    raise ValueError(f"Unsupported file type: {filename}. Use .pdf, .docx, or .txt")


def _guess_candidate_name(text: str) -> str | None:
    for line in text.splitlines()[:8]:
        line = line.strip()
        if not line or "@" in line or any(ch.isdigit() for ch in line):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
            return line
    return None


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+|\n|•|‣|·", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def _find_resume_skills(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for skill in settings.SKILL_VOCABULARY:
        variant = re.escape(skill).replace(r"\.", r"[.\s]?").replace(r"\ ", r"[\s/-]?")
        if re.search(rf"\b{variant}\b", text_lower):
            found.append(skill)
    return found


def _extract_experience_years(text: str) -> float | None:
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b(?:\s*of)?\s*(?:experience)?",
        text, re.IGNORECASE,
    )
    if matches:
        return max(float(m) for m in matches)
    return None


def parse_resume(filename: str, file_bytes: bytes) -> ParsedResume:
    raw_text = extract_text(filename, file_bytes)
    sentences = _split_sentences(raw_text)
    word_count = len(raw_text.split())

    return ParsedResume(
        filename=filename,
        raw_text=raw_text,
        sentences=sentences,
        candidate_name=_guess_candidate_name(raw_text),
        detected_skills=_find_resume_skills(raw_text),
        detected_experience_years=_extract_experience_years(raw_text),
        word_count=word_count,
    )


# ════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS  —  sentence-transformers w/ automatic TF-IDF fallback
# ════════════════════════════════════════════════════════════════════════
class EmbeddingBackend(Protocol):
    name: str
    def fit(self, corpus: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class TfidfBackend:
    name = "tfidf"

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, corpus: list[str]) -> None:
        corpus = [c if c.strip() else "empty" for c in corpus] or ["empty"]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._vectorizer.fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None:
            self.fit(texts)
        texts = [t if t.strip() else "empty" for t in texts]
        return self._vectorizer.transform(texts).toarray()  # type: ignore[union-attr]


class SentenceTransformerBackend:
    name = "sentence-transformers"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)

    def fit(self, corpus: list[str]) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        texts = [t if t.strip() else "empty" for t in texts]
        return self._model.encode(texts, show_progress_bar=False)


_embedder: EmbeddingBackend | None = None


def get_embedder() -> EmbeddingBackend:
    global _embedder
    if _embedder is not None:
        return _embedder

    choice = settings.EMBEDDING_BACKEND

    if choice == "tfidf":
        _embedder = TfidfBackend()
        return _embedder

    if choice in ("sentence-transformers", "auto"):
        try:
            _embedder = SentenceTransformerBackend()
            return _embedder
        except Exception:
            if choice == "sentence-transformers":
                raise
            _embedder = TfidfBackend()
            return _embedder

    raise ValueError(f"Unknown EMBEDDING_BACKEND: {choice}")


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cosine_similarity(a, b)


# ════════════════════════════════════════════════════════════════════════
#  SCORING  —  multi-signal candidate scoring
# ════════════════════════════════════════════════════════════════════════
MUST_HAVE_WEIGHT = 0.7
NICE_TO_HAVE_WEIGHT = 0.3


def _best_sentence_match(
    skill: str, resume: ParsedResume, embedder: EmbeddingBackend
) -> tuple[float, str | None]:
    if skill in resume.detected_skills:
        for sentence in resume.sentences:
            if skill.lower().replace(".", "").replace(" ", "") in (
                sentence.lower().replace(".", "").replace(" ", "")
            ):
                return 1.0, sentence
        return 1.0, resume.sentences[0] if resume.sentences else None

    if not resume.sentences:
        return 0.0, None

    skill_vec = embedder.encode([skill])
    sentence_vecs = embedder.encode(resume.sentences)
    sims = cosine_sim_matrix(skill_vec, sentence_vecs)[0]
    sims = np.clip(sims, 0.0, 1.0)
    best_idx = int(np.argmax(sims))
    return float(sims[best_idx]), resume.sentences[best_idx]


def _score_skills(
    skills: list[str], resume: ParsedResume, embedder: EmbeddingBackend
) -> list[SkillEvidence]:
    evidence = []
    for skill in skills:
        sim, sentence = _best_sentence_match(skill, resume, embedder)
        matched = sim >= settings.SKILL_MATCH_THRESHOLD
        evidence.append(
            SkillEvidence(
                skill=skill, matched=matched, similarity=round(sim, 3),
                evidence_sentence=sentence if matched else None,
            )
        )
    return evidence


def _skill_match_score(
    must_have_evidence: list[SkillEvidence], nice_to_have_evidence: list[SkillEvidence]
) -> float:
    must_total = len(must_have_evidence) * MUST_HAVE_WEIGHT
    nice_total = len(nice_to_have_evidence) * NICE_TO_HAVE_WEIGHT
    denom = must_total + nice_total
    if denom == 0:
        return 0.5
    must_matched = sum(e.matched for e in must_have_evidence) * MUST_HAVE_WEIGHT
    nice_matched = sum(e.matched for e in nice_to_have_evidence) * NICE_TO_HAVE_WEIGHT
    return (must_matched + nice_matched) / denom


def _experience_score(candidate_years: float | None, min_required: float | None) -> float:
    if candidate_years is None:
        return 0.3
    if min_required is None or min_required <= 0:
        return 0.8
    ratio = candidate_years / min_required
    return float(min(1.0, max(0.0, ratio)))


def _semantic_score(jd_text: str, resume_text: str, embedder: EmbeddingBackend) -> float:
    vecs = embedder.encode([jd_text, resume_text])
    sim = cosine_sim_matrix(vecs[0:1], vecs[1:2])[0][0]
    return float(np.clip(sim, 0.0, 1.0))


def score_candidate(
    parsed_jd: ParsedJD, resume: ParsedResume, embedder: EmbeddingBackend
) -> tuple[ScoreBreakdown, list[SkillEvidence], list[SkillEvidence]]:
    must_have_evidence = _score_skills(parsed_jd.must_have_skills, resume, embedder)
    nice_to_have_evidence = _score_skills(parsed_jd.nice_to_have_skills, resume, embedder)

    skill_score = _skill_match_score(must_have_evidence, nice_to_have_evidence)
    experience_score = _experience_score(
        resume.detected_experience_years, parsed_jd.min_experience_years
    )
    semantic_score = _semantic_score(parsed_jd.raw_text, resume.raw_text, embedder)

    final = (
        skill_score * settings.WEIGHT_SKILL_MATCH
        + experience_score * settings.WEIGHT_EXPERIENCE
        + semantic_score * settings.WEIGHT_SEMANTIC
    ) * 100

    breakdown = ScoreBreakdown(
        skill_match_score=round(skill_score * 100, 1),
        experience_score=round(experience_score * 100, 1),
        semantic_score=round(semantic_score * 100, 1),
        final_score=round(final, 1),
    )
    return breakdown, must_have_evidence, nice_to_have_evidence


# ════════════════════════════════════════════════════════════════════════
#  VALIDATION  —  data-quality / suspicious-profile checks
# ════════════════════════════════════════════════════════════════════════
def _content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_resumes(resumes: list[ParsedResume]) -> list[DataQualityFlag]:
    hashes = [_content_hash(r.raw_text) for r in resumes]
    flags: list[DataQualityFlag] = []

    for i, resume in enumerate(resumes):
        notes: list[str] = []
        is_sparse = resume.word_count < settings.MIN_RESUME_WORDS
        if is_sparse:
            notes.append(
                f"Only {resume.word_count} words extracted — resume may be "
                "scanned/image-based, malformed, or genuinely sparse."
            )

        is_duplicate = hashes[i] in hashes[:i]
        if is_duplicate:
            dupe_of = resumes[hashes[:i].index(hashes[i])].filename
            notes.append(f"Near-identical content to '{dupe_of}'.")

        if not resume.detected_skills:
            notes.append("No skills from the vocabulary were detected at all.")

        flags.append(
            DataQualityFlag(
                is_sparse=is_sparse, is_duplicate=is_duplicate,
                word_count=resume.word_count, notes=notes,
            )
        )
    return flags


# ════════════════════════════════════════════════════════════════════════
#  EXPLAIN  —  evidence-grounded human-readable summaries
# ════════════════════════════════════════════════════════════════════════
def missing_must_haves(must_have_evidence: list[SkillEvidence]) -> list[str]:
    return [e.skill for e in must_have_evidence if not e.matched]


def build_summary(
    candidate_name: str | None, score: ScoreBreakdown,
    must_have_evidence: list[SkillEvidence], nice_to_have_evidence: list[SkillEvidence],
    quality: DataQualityFlag,
) -> str:
    name = candidate_name or "This candidate"
    matched_must = [e.skill for e in must_have_evidence if e.matched]
    missing = missing_must_haves(must_have_evidence)
    matched_nice = [e.skill for e in nice_to_have_evidence if e.matched]

    parts = [f"{name} scores {score.final_score}/100."]

    if matched_must:
        parts.append(f"Matches {len(matched_must)}/{len(must_have_evidence)} must-have skills: "
                      f"{', '.join(matched_must)}.")
    elif must_have_evidence:
        parts.append("Matches none of the must-have skills.")

    if missing:
        parts.append(f"Missing: {', '.join(missing)}.")

    if matched_nice:
        parts.append(f"Also brings {len(matched_nice)} nice-to-have skill(s): "
                      f"{', '.join(matched_nice)}.")

    if score.final_score < settings.WEAK_MATCH_THRESHOLD:
        parts.append("Overall a weak match for this role.")

    if quality.is_sparse:
        parts.append("⚠ Resume content is sparse — review manually.")
    if quality.is_duplicate:
        parts.append("⚠ Flagged as a likely duplicate submission.")

    return " ".join(parts)


# ════════════════════════════════════════════════════════════════════════
#  RANKING  —  orchestrates the full pipeline
# ════════════════════════════════════════════════════════════════════════
def _candidate_id(filename: str, raw_text: str) -> str:
    digest = hashlib.sha1(f"{filename}:{raw_text[:200]}".encode("utf-8")).hexdigest()
    return digest[:10]


def rank_candidates(jd_text: str, resume_files: list[tuple[str, bytes]]) -> RankResponse:
    parsed_jd: ParsedJD = parse_jd(jd_text)
    resumes: list[ParsedResume] = [
        parse_resume(filename, file_bytes) for filename, file_bytes in resume_files
    ]

    embedder = get_embedder()
    corpus = [parsed_jd.raw_text]
    corpus += parsed_jd.must_have_skills + parsed_jd.nice_to_have_skills
    for r in resumes:
        corpus.append(r.raw_text)
        corpus += r.sentences
    embedder.fit(corpus)

    quality_flags = validate_resumes(resumes)

    results: list[CandidateResult] = []
    for resume, quality in zip(resumes, quality_flags):
        score, must_have_evidence, nice_to_have_evidence = score_candidate(
            parsed_jd, resume, embedder
        )
        summary = build_summary(
            resume.candidate_name, score, must_have_evidence, nice_to_have_evidence, quality
        )
        results.append(
            CandidateResult(
                candidate_id=_candidate_id(resume.filename, resume.raw_text),
                filename=resume.filename,
                candidate_name=resume.candidate_name,
                detected_experience_years=resume.detected_experience_years,
                score=score,
                must_have_evidence=must_have_evidence,
                nice_to_have_evidence=nice_to_have_evidence,
                missing_must_haves=missing_must_haves(must_have_evidence),
                data_quality=quality,
                summary=summary,
            )
        )

    results.sort(key=lambda c: c.score.final_score, reverse=True)

    return RankResponse(
        parsed_jd=parsed_jd, candidates=results, embedding_backend=embedder.name,
    )


# ════════════════════════════════════════════════════════════════════════
#  FRONTEND  —  single-page dashboard, served inline (no separate file)
# ════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>RankSense — JD-to-Candidate Ranking</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root { --navy:#1b1740; --violet:#5b3fd6; --orange:#ff6a39; --ink:#211d45; --muted:#6b6890; --card:#f6f5fb; --line:#e4e1f2; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif; color:var(--ink); background:#fff; }
  header { background:var(--navy); color:#fff; padding:28px 40px; }
  header h1 { margin:0 0 4px 0; font-size:26px; }
  header p { margin:0; color:#c8bef2; font-size:14px; }
  main { max-width:1100px; margin:0 auto; padding:32px 24px 80px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  @media (max-width:800px) { .grid { grid-template-columns:1fr; } }
  label { display:block; font-weight:600; margin-bottom:6px; font-size:14px; }
  textarea, input[type=file] { width:100%; border:1px solid var(--line); border-radius:8px; padding:10px; font-family:inherit; font-size:14px; }
  textarea { height:220px; resize:vertical; }
  .hint { color:var(--muted); font-size:12px; margin-top:6px; }
  button { background:var(--orange); color:#fff; border:none; border-radius:8px; padding:12px 24px; font-size:15px; font-weight:600; cursor:pointer; margin-top:20px; }
  button:disabled { background:#cbb; cursor:wait; opacity:0.6; }
  #status { margin-top:12px; font-size:14px; color:var(--muted); }
  .results { margin-top:40px; }
  .jd-summary { background:var(--card); border-radius:10px; padding:18px 22px; margin-bottom:28px; }
  .jd-summary h3 { margin:0 0 10px 0; }
  .pill { display:inline-block; background:#fff; border:1px solid var(--line); border-radius:999px; padding:4px 12px; font-size:12px; margin:2px 4px 2px 0; }
  .card { border:1px solid var(--line); border-radius:10px; padding:20px 22px; margin-bottom:18px; }
  .card-head { display:flex; justify-content:space-between; align-items:baseline; gap:12px; }
  .card-head h4 { margin:0; font-size:17px; }
  .score { font-size:26px; font-weight:800; color:var(--navy); }
  .score.weak { color:#b3463b; }
  .meta { color:var(--muted); font-size:13px; margin-top:2px; }
  .summary { margin-top:10px; font-size:14px; line-height:1.5; }
  details { margin-top:12px; }
  summary { cursor:pointer; font-size:13px; color:var(--violet); font-weight:600; }
  .breakdown { display:flex; gap:18px; margin-top:10px; flex-wrap:wrap; }
  .bd-item { background:var(--card); border-radius:8px; padding:8px 14px; font-size:12px; }
  .bd-item b { display:block; font-size:16px; color:var(--navy); }
  .evidence-list { margin-top:10px; font-size:13px; }
  .evidence-list .ok { color:#1d7a4c; }
  .evidence-list .miss { color:#b3463b; }
  .quote { color:var(--muted); font-style:italic; }
  .flag { color:#b3463b; font-weight:600; font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<header>
  <h1>RankSense</h1>
  <p>Explainable AI engine for JD-to-Candidate matching &amp; ranking</p>
</header>
<main>
  <div class="grid">
    <div>
      <label for="jd">Job Description</label>
      <textarea id="jd" placeholder="Paste the job description here..."></textarea>
      <div class="hint">Tip: include phrasing like "5+ years" and mark optional skills with words like "preferred" or "nice to have" for best extraction.</div>
    </div>
    <div>
      <label for="resumes">Candidate Resumes</label>
      <input id="resumes" type="file" multiple accept=".pdf,.docx,.txt" />
      <div class="hint">Upload one or more .pdf, .docx, or .txt resumes.</div>
    </div>
  </div>
  <button id="rankBtn">Rank Candidates</button>
  <div id="status"></div>
  <div class="results" id="results"></div>
</main>
<script>
const jdEl = document.getElementById('jd');
const resumesEl = document.getElementById('resumes');
const btn = document.getElementById('rankBtn');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');

btn.addEventListener('click', async () => {
  const jdText = jdEl.value.trim();
  const files = resumesEl.files;
  if (jdText.length < 20) { statusEl.textContent = 'Please paste a longer job description (20+ characters).'; return; }
  if (!files.length) { statusEl.textContent = 'Please upload at least one resume.'; return; }

  const form = new FormData();
  form.append('jd_text', jdText);
  for (const f of files) form.append('resumes', f);

  btn.disabled = true;
  statusEl.textContent = 'Parsing JD, scoring candidates...';
  resultsEl.innerHTML = '';

  try {
    const res = await fetch('/api/rank', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    render(data);
    statusEl.textContent = `Done — embedding backend: ${data.embedding_backend}`;
  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
});

function pill(text) { return `<span class="pill">${escapeHtml(text)}</span>`; }
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function evidenceRow(e) {
  const cls = e.matched ? 'ok' : 'miss';
  const mark = e.matched ? '✓' : '✗';
  let line = `<div class="${cls}">${mark} <b>${escapeHtml(e.skill)}</b> (similarity: ${e.similarity})`;
  if (e.matched && e.evidence_sentence) line += `<div class="quote">"${escapeHtml(e.evidence_sentence)}"</div>`;
  line += `</div>`;
  return line;
}
function render(data) {
  const jd = data.parsed_jd;
  resultsEl.innerHTML = `
    <div class="jd-summary">
      <h3>${escapeHtml(jd.title || 'Parsed Job Description')}</h3>
      <div><b>Must-have:</b> ${jd.must_have_skills.map(pill).join('') || '<span class="hint">none detected</span>'}</div>
      <div style="margin-top:8px;"><b>Nice-to-have:</b> ${jd.nice_to_have_skills.map(pill).join('') || '<span class="hint">none detected</span>'}</div>
      <div style="margin-top:8px;" class="meta">Min. experience: ${jd.min_experience_years ?? 'not specified'} years &nbsp;•&nbsp; Seniority: ${jd.seniority ?? 'not specified'}</div>
    </div>
  ` + data.candidates.map(candidateCard).join('');
}
function candidateCard(c, idx) {
  const weak = c.score.final_score < 40;
  const flags = [];
  if (c.data_quality.is_sparse) flags.push('⚠ Sparse resume content');
  if (c.data_quality.is_duplicate) flags.push('⚠ Likely duplicate submission');
  return `
    <div class="card">
      <div class="card-head">
        <div>
          <h4>#${idx + 1} — ${escapeHtml(c.candidate_name || c.filename)}</h4>
          <div class="meta">${escapeHtml(c.filename)} • Experience detected: ${c.detected_experience_years ?? 'unknown'} yrs</div>
        </div>
        <div class="score ${weak ? 'weak' : ''}">${c.score.final_score}<span style="font-size:14px;">/100</span></div>
      </div>
      <div class="summary">${escapeHtml(c.summary)}</div>
      <div class="breakdown">
        <div class="bd-item"><b>${c.score.skill_match_score}</b>Skill Match</div>
        <div class="bd-item"><b>${c.score.experience_score}</b>Experience Fit</div>
        <div class="bd-item"><b>${c.score.semantic_score}</b>Semantic Fit</div>
      </div>
      ${flags.length ? `<div class="flag">${flags.join(' &nbsp;•&nbsp; ')}</div>` : ''}
      <details>
        <summary>Show evidence (must-have &amp; nice-to-have skills)</summary>
        <div class="evidence-list">
          <b>Must-have:</b>
          ${c.must_have_evidence.map(evidenceRow).join('') || '<div class="hint">none required</div>'}
          <b style="display:block; margin-top:10px;">Nice-to-have:</b>
          ${c.nice_to_have_evidence.map(evidenceRow).join('') || '<div class="hint">none listed</div>'}
        </div>
      </details>
    </div>
  `;
}
</script>
</body>
</html>
"""


# ════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="RankSense API",
    description="Explainable AI engine for JD-to-candidate matching & ranking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> str:
    return DASHBOARD_HTML


@app.post("/api/parse-jd", response_model=ParsedJD)
def api_parse_jd(payload: JDParseRequest) -> ParsedJD:
    return parse_jd(payload.jd_text)


@app.post("/api/rank", response_model=RankResponse)
async def api_rank(
    jd_text: str = Form(..., min_length=20),
    resumes: list[UploadFile] = File(...),
) -> RankResponse:
    if not resumes:
        raise HTTPException(status_code=400, detail="Upload at least one resume.")

    files: list[tuple[str, bytes]] = []
    for upload in resumes:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{upload.filename}'. "
                f"Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        content = await upload.read()
        files.append((upload.filename or "resume", content))

    try:
        return rank_candidates(jd_text, files)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {exc}") from exc


# ════════════════════════════════════════════════════════════════════════
#  BUILT-IN SAMPLE DATA — used by --demo and by the test suite below
# ════════════════════════════════════════════════════════════════════════
SAMPLE_JD = """Senior Backend Engineer

We are looking for a Senior Backend Engineer with 5+ years of experience
to join our platform team.

Requirements:
- 5+ years of professional software engineering experience
- Strong proficiency in Python and SQL
- Experience with AWS, Docker, and Kubernetes
- Solid understanding of REST API design and microservices architecture
- Experience with CI/CD pipelines and Git

Nice to have:
- Familiarity with machine learning or data engineering workflows
- Experience with GraphQL is a plus
- Exposure to Terraform is a bonus

We value strong communication and leadership skills.
"""

SAMPLE_RESUMES: dict[str, bytes] = {
    "asha_mehta.txt": b"""Asha Mehta
6 years of experience building backend services in Python and SQL.
Designed and built REST APIs and microservices using Python and FastAPI.
Deployed services on AWS using Docker and Kubernetes.
Set up CI/CD pipelines with Git-based workflows.
Mentored junior engineers and led architecture reviews.
""",
    "rohan_verma.txt": b"""Rohan Verma
3 years of experience as a software developer.
Built internal tools using Python and basic SQL queries.
Worked with Docker for local development environments.
Skills: Python, SQL, Docker, Java, Git, communication.
""",
    "priya_nair.txt": b"""Priya Nair
4 years of experience in digital marketing, SEO, and SEM campaigns.
Used Excel and Power BI for campaign performance reporting.
Coordinated with sales teams using Salesforce CRM.
""",
}


def run_cli_demo() -> None:
    print("=" * 70)
    print("RankSense CLI demo — ranking built-in sample resumes against a sample JD")
    print("=" * 70)

    result = rank_candidates(SAMPLE_JD, list(SAMPLE_RESUMES.items()))

    print(f"\nPARSED JD: {result.parsed_jd.title}")
    print(f"Must-have skills:    {', '.join(result.parsed_jd.must_have_skills)}")
    print(f"Nice-to-have skills: {', '.join(result.parsed_jd.nice_to_have_skills)}")
    print(f"Min experience:      {result.parsed_jd.min_experience_years} years")
    print(f"Embedding backend:   {result.embedding_backend}")
    print("-" * 70)

    for i, c in enumerate(result.candidates, start=1):
        print(f"\n#{i}  {c.candidate_name or c.filename}  —  {c.score.final_score}/100")
        print(f"     skill={c.score.skill_match_score}  "
              f"experience={c.score.experience_score}  "
              f"semantic={c.score.semantic_score}")
        print(f"     {c.summary}")
        if c.missing_must_haves:
            print(f"     Missing: {', '.join(c.missing_must_haves)}")

    print("\n" + "=" * 70)


# ════════════════════════════════════════════════════════════════════════
#  TESTS  —  run with: pytest ranksense.py
#  (pytest auto-discovers test_* functions in any importable file)
# ════════════════════════════════════════════════════════════════════════
def test_jd_parser_extracts_must_have_and_nice_to_have():
    os.environ["EMBEDDING_BACKEND"] = "tfidf"
    parsed = parse_jd_rule_based(SAMPLE_JD)
    assert "python" in parsed.must_have_skills
    assert "aws" in parsed.must_have_skills
    assert "machine learning" in parsed.nice_to_have_skills
    assert "machine learning" not in parsed.must_have_skills
    assert parsed.min_experience_years == 5.0


def test_resume_parser_extracts_skills_and_experience():
    resume = parse_resume("asha.txt", SAMPLE_RESUMES["asha_mehta.txt"])
    assert "python" in resume.detected_skills
    assert "aws" in resume.detected_skills
    assert resume.detected_experience_years == 6.0


def test_ranking_orders_strong_match_above_weak_match():
    global _embedder
    _embedder = TfidfBackend()
    result = rank_candidates(
        SAMPLE_JD,
        [("asha.txt", SAMPLE_RESUMES["asha_mehta.txt"]),
         ("priya.txt", SAMPLE_RESUMES["priya_nair.txt"])],
    )
    assert result.candidates[0].filename == "asha.txt"
    assert result.candidates[0].score.final_score > result.candidates[1].score.final_score


def test_strong_match_has_evidence_for_matched_skills():
    global _embedder
    _embedder = TfidfBackend()
    result = rank_candidates(SAMPLE_JD, [("asha.txt", SAMPLE_RESUMES["asha_mehta.txt"])])
    candidate = result.candidates[0]
    python_evidence = next(e for e in candidate.must_have_evidence if e.skill == "python")
    assert python_evidence.matched is True
    assert python_evidence.evidence_sentence is not None


def test_sparse_resume_is_flagged():
    resume = parse_resume("sparse.txt", b"J. Doe\nLooking for a job.")
    flags = validate_resumes([resume])
    assert flags[0].is_sparse is True


def test_duplicate_resume_is_flagged():
    content = SAMPLE_RESUMES["asha_mehta.txt"]
    resume_a = parse_resume("a.txt", content)
    resume_b = parse_resume("b.txt", content)
    flags = validate_resumes([resume_a, resume_b])
    assert flags[0].is_duplicate is False
    assert flags[1].is_duplicate is True


def test_missing_must_have_skills_are_reported():
    global _embedder
    _embedder = TfidfBackend()
    result = rank_candidates(SAMPLE_JD, [("priya.txt", SAMPLE_RESUMES["priya_nair.txt"])])
    candidate = result.candidates[0]
    assert "python" in candidate.missing_must_haves
    assert candidate.score.final_score < 50


# ════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RankSense — single-file edition")
    parser.add_argument("--demo", action="store_true", help="Run the CLI demo and exit")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web server")
    args = parser.parse_args()

    if args.demo:
        run_cli_demo()
        sys.exit(0)

    import uvicorn
    print(f"Starting RankSense at http://localhost:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
