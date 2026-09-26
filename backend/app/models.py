from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ResumeUploadResponse(BaseModel):
    session_id: str
    filename: str
    total_characters: int
    chunks_stored: int
    message: str

class JobDescriptionRequest(BaseModel):
    session_id: str = Field(default="default", description="Session or user identifier")
    job_description: str = Field(..., description="The complete text of the job description")

class SuitabilityScoreBreakdown(BaseModel):
    overall_score: int = Field(..., ge=0, le=100, description="Overall match score from 0 to 100")
    fit_level: str = Field(..., description="'Strong Fit', 'Moderate Fit', or 'Low Fit'")
    skills_match_score: int = Field(..., ge=0, le=100)
    experience_match_score: int = Field(..., ge=0, le=100)
    education_match_score: int = Field(..., ge=0, le=100)
    key_strengths: List[str] = Field(default_factory=list, description="Strengths matching the JD")
    missing_skills_and_gaps: List[str] = Field(default_factory=list, description="Missing requirements/gaps")
    critical_missing_keywords: List[str] = Field(default_factory=list, description="ATS keywords not found in resume")
    actionable_recommendations: List[str] = Field(default_factory=list, description="How to improve or tailor resume")
    summary: str = Field(..., description="Executive summary of candidate suitability")

class EvaluateRequest(BaseModel):
    session_id: str = Field(default="default", description="Session or user identifier")
    job_description: str = Field(..., description="Job description text")

class RetrievedEvidence(BaseModel):
    chunk_id: str
    content: str
    similarity_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvaluateResponse(BaseModel):
    session_id: str
    suitability: SuitabilityScoreBreakdown
    retrieved_evidence: List[RetrievedEvidence]

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str

class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="Session or user identifier")
    message: str = Field(..., description="User question or instruction")
    job_description: Optional[str] = Field(None, description="Optional override or update for job description")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Conversation history")

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    retrieved_sources: List[RetrievedEvidence] = Field(default_factory=list)

class SystemStatusResponse(BaseModel):
    status: str
    openai_configured: bool
    chroma_db_path: str
    model_name: str
