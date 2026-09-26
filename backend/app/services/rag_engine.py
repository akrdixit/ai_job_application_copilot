import json
import logging
import re
from typing import List, Dict, Any, Optional, Generator
from openai import OpenAI

from backend.app.config import OPENAI_API_KEY, OPENAI_MODEL
from backend.app.models import SuitabilityScoreBreakdown, RetrievedEvidence, ChatMessage
from backend.app.services.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Resumes are small, so for match analysis we send the whole thing instead of top-k
# chunks (partial evidence makes the model report skills as "missing" when they aren't).
FULL_RESUME_CHAR_LIMIT = 24000

MATCH_INTENT_PATTERN = re.compile(
    r"\b(match|matching|fit|fits|suitable|suitability|qualified|qualify|eligible|"
    r"good (candidate|fit)|right for|should i apply|chances|compare|score|gap|gaps|"
    r"missing|ats|shortlist)\b",
    re.IGNORECASE,
)

JD_MARKERS = [
    "responsibilities", "requirements", "qualifications", "job description",
    "about the role", "what you'll do", "what you will do", "who you are",
    "must have", "nice to have", "preferred", "years of experience",
    "we are looking for", "we're looking for", "you will", "benefits",
    "apply now", "job type", "location:", "salary",
]

SYSTEM_PROMPT = """You are an AI Job Application Assistant and Career Coach.

You help candidates evaluate their resume against a job description, identify strengths and gaps, tailor their resume, draft cover letters, and prepare for interviews.

If you have resume context provided below, use it to ground your answers. If no resume context is available, respond helpfully based on general career advice and gently invite the user to attach their resume.

Always respond in clean markdown. Be direct, specific, and constructive."""

MATCH_ANALYSIS_PROMPT = """The user wants to know whether their resume matches the job description. Both are provided above. Analyze them like an experienced recruiter and ATS screener:

1. Extract the JD's requirements and split them into **must-haves** and **nice-to-haves**. Note the seniority level and required years of experience.
2. For each requirement, check the resume for evidence. Treat synonyms, related tools, and clearly transferable experience as partial matches (e.g. PostgreSQL satisfies "SQL", Flask partially satisfies "Django"). Never invent experience that is not in the resume.
3. Weigh must-haves far more heavily than nice-to-haves. Missing a hard requirement (years of experience, required degree/certification, core technology) should significantly lower the verdict.

Respond in this markdown structure:

**Verdict:** <✅ Strong Match | 🟡 Partial Match | ❌ Weak Match> — **<score>/100**

<one or two sentence bottom line: should they apply?>

| Requirement | Type | Status | Evidence from resume |
|---|---|---|---|
(one row per key requirement; Type = Must-have / Nice-to-have; Status = ✅ Met / 🟡 Partial / ❌ Missing)

**Strengths:** bullet list
**Gaps & risks:** bullet list
**Missing ATS keywords:** comma-separated list
**How to close the gap:** 3-5 specific, actionable resume edits or talking points

Scoring: 80-100 = Strong Match, 60-79 = Partial Match, below 60 = Weak Match."""

MISSING_RESUME_NOTE = (
    "The user is asking whether they match a job, but no resume has been uploaded for this session. "
    "Give a brief summary of what the job requires, then ask them to upload their resume so you can do a real match analysis."
)

MISSING_JD_NOTE = (
    "The user is asking whether their resume matches a job, but no job description has been provided. "
    "Ask them to paste the job description. Do not guess a match score without it."
)


def looks_like_job_description(text: str) -> bool:
    """Heuristic: long text containing several typical JD section markers."""
    if not text or len(text) < 300:
        return False
    lowered = text.lower()
    return sum(1 for marker in JD_MARKERS if marker in lowered) >= 2


def is_match_query(message: str) -> bool:
    return bool(message and MATCH_INTENT_PATTERN.search(message))


class RAGEngine:
    """Orchestrates RAG retrieval against ChromaDB and evaluations using OpenAI."""

    def __init__(self, vector_store: VectorStoreManager, api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.api_key = api_key or OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not set. Please provide a valid OpenAI API key in .env or via settings.")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("sk-"))

    def _get_full_resume_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns every resume chunk in order, or [] if the resume is too large to send whole."""
        chunks = self.vector_store.get_all_chunks(session_id)
        if sum(len(c["content"]) for c in chunks) > FULL_RESUME_CHAR_LIMIT:
            return []
        return chunks

    def _format_resume(self, chunks: List[Dict[str, Any]]) -> str:
        return "\n\n---\n\n".join([
            f"[{c['metadata'].get('section', 'Resume')}]\n{c['content']}"
            for c in chunks
        ])

    def _build_context(self, session_id: str, message: str, job_description: Optional[str]) -> str:
        """Retrieves resume chunks and builds a context block for the LLM.

        Switches into match-analysis mode when the user asks about fit, or pastes a JD
        directly into the chat.
        """
        jd_in_message = looks_like_job_description(message)
        if jd_in_message and not job_description:
            job_description = message
        match_mode = jd_in_message or is_match_query(message)
        has_resume = self.vector_store.has_resume(session_id)

        parts = []
        if job_description:
            parts.append(f"### Job Description:\n{job_description}")

        if has_resume:
            chunks = self._get_full_resume_chunks(session_id) if match_mode else []
            if not chunks:
                query = f"{message} {(job_description or '')[:200]}"
                chunks = self.vector_store.search_relevant_chunks(session_id, query, top_k=8 if match_mode else 5)
            if chunks:
                parts.append(f"### Resume Context:\n{self._format_resume(chunks)}")

        if match_mode:
            if not has_resume:
                parts.append(MISSING_RESUME_NOTE)
            elif not job_description:
                parts.append(MISSING_JD_NOTE)
            else:
                parts.append(MATCH_ANALYSIS_PROMPT)

        return "\n\n".join(parts)

    def _build_messages(self, context: str, history: Optional[List[ChatMessage]], message: str) -> List[Dict[str, str]]:
        """Builds the messages list for the OpenAI API call."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            messages.append({"role": "system", "content": context})

        if history:
            for h in history[-8:]:
                role = h.role if hasattr(h, "role") else h.get("role", "user")
                content = h.content if hasattr(h, "content") else h.get("content", "")
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})
        return messages

    def evaluate_fit(self, session_id: str, job_description: str) -> Dict[str, Any]:
        """
        Retrieves relevant resume chunks from ChromaDB and performs an in-depth
        suitability evaluation against the provided job description.
        """
        if not self.vector_store.has_resume(session_id):
            raise ValueError(f"No resume found for session '{session_id}'. Please upload a resume first.")

        # Prefer the whole resume so nothing is wrongly flagged as missing
        retrieved_chunks = self._get_full_resume_chunks(session_id)
        for c in retrieved_chunks:
            c["similarity_score"] = 1.0

        if not retrieved_chunks:
            # Resume too large: single-request batch query across skills, experience, and education
            queries = [
                f"Core technical skills, competencies, tools, programming languages required: {job_description[:300]}",
                f"Work experience, job titles, responsibilities, projects, leadership: {job_description[:300]}",
                "Education degree university major certifications credentials"
            ]
            retrieved_chunks = self.vector_store.search_batch_chunks(session_id, queries, top_k=6)

        # Format context for prompt
        context_str = "\n\n---\n\n".join([
            f"[Chunk ID: {c['chunk_id']} | Section: {c['metadata'].get('section', 'General')}]\n{c['content']}"
            for c in retrieved_chunks
        ])

        eval_system = (
            "You are an expert AI Career Coach, technical recruiter, and ATS Specialist.\n"
            "Evaluate whether the candidate's resume is suitable for the given job description.\n"
            "Base your evaluation strictly on the provided resume evidence; never assume experience that isn't written.\n\n"
            "Method:\n"
            "1. Extract the JD requirements and classify each as must-have or nice-to-have. Note seniority and years of experience required.\n"
            "2. Match each requirement against the resume. Count synonyms, related tools, and transferable experience as partial matches.\n"
            "3. Weigh must-haves much more heavily than nice-to-haves. A missing hard requirement (years of experience, "
            "required degree/certification, core technology) must pull the overall score down noticeably.\n"
            "4. The overall_score should reflect how likely this candidate is to pass screening, not an average of sub-scores.\n\n"
            "Return a valid JSON object with exactly this structure:\n"
            "{\n"
            '  "overall_score": <integer 0-100>,\n'
            '  "fit_level": "<Strong Fit | Moderate Fit | Low Fit>",\n'
            '  "skills_match_score": <integer 0-100>,\n'
            '  "experience_match_score": <integer 0-100>,\n'
            '  "education_match_score": <integer 0-100>,\n'
            '  "key_strengths": ["specific strength, citing resume evidence"],\n'
            '  "missing_skills_and_gaps": ["gap, marked (must-have) or (nice-to-have)"],\n'
            '  "critical_missing_keywords": ["..."],\n'
            '  "actionable_recommendations": ["..."],\n'
            '  "summary": "2-3 sentence executive summary including a clear apply / don\'t apply recommendation"\n'
            "}\n\n"
            "Scoring: 80-100 = Strong Fit, 60-79 = Moderate Fit, below 60 = Low Fit."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": eval_system},
                {"role": "user", "content": f"### Job Description:\n{job_description}\n\n### Resume Evidence:\n{context_str}\n\nEvaluate and return JSON."}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        parsed_data = self._normalize_scores(json.loads(response.choices[0].message.content or "{}"))

        evidence_objects = [
            RetrievedEvidence(
                chunk_id=c["chunk_id"],
                content=c["content"],
                similarity_score=c.get("similarity_score"),
                metadata=c.get("metadata", {})
            )
            for c in retrieved_chunks
        ]

        return {
            "session_id": session_id,
            "suitability": SuitabilityScoreBreakdown(**parsed_data),
            "retrieved_evidence": evidence_objects
        }

    @staticmethod
    def _normalize_scores(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clamps scores to 0-100 and keeps fit_level consistent with overall_score."""
        for key in ("overall_score", "skills_match_score", "experience_match_score", "education_match_score"):
            try:
                data[key] = max(0, min(100, int(round(float(data.get(key, 0))))))
            except (TypeError, ValueError):
                data[key] = 0
        score = data["overall_score"]
        data["fit_level"] = "Strong Fit" if score >= 80 else "Moderate Fit" if score >= 60 else "Low Fit"
        data.setdefault("summary", "")
        return data

    def chat(self, session_id: str, message: str, job_description: Optional[str] = None, history: Optional[List[ChatMessage]] = None) -> Dict[str, Any]:
        """Handles conversational Q&A using RAG."""
        context = self._build_context(session_id, message, job_description)
        messages = self._build_messages(context, history, message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5
        )

        return {
            "session_id": session_id,
            "reply": response.choices[0].message.content or "",
            "retrieved_sources": []
        }

    def chat_stream(self, session_id: str, message: str, job_description: Optional[str] = None, history: Optional[List[ChatMessage]] = None) -> Generator[str, None, None]:
        """Streams conversational RAG tokens chunk-by-chunk."""
        context = self._build_context(session_id, message, job_description)
        messages = self._build_messages(context, history, message)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
