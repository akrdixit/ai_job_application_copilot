import os
import logging
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.app.config import (
    OPENAI_API_KEY, OPENAI_MODEL, CHROMA_DB_PATH, BACKEND_HOST, BACKEND_PORT
)
from backend.app.models import (
    ResumeUploadResponse, EvaluateRequest, EvaluateResponse,
    ChatRequest, ChatResponse, SystemStatusResponse
)
from backend.app.services.parser import ResumeParser
from backend.app.services.vector_store import VectorStoreManager
from backend.app.services.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("job_chatbot_api")

app = FastAPI(
    title="AI Job Application RAG Chatbot API",
    description="Backend API for vector-embedded resume analysis, job suitability scoring, and conversational RAG",
    version="1.0.0"
)

# Enable CORS for local development and Chainlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store and RAG engine instances
vector_store = VectorStoreManager(persist_directory=CHROMA_DB_PATH)
rag_engine = RAGEngine(vector_store=vector_store)

# In-memory store for session target job descriptions
session_jds: Dict[str, str] = {}


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "AI Job Application Chatbot API is operational.",
        "docs_url": "/docs",
        "openai_configured": rag_engine.is_configured()
    }


@app.get("/api/status", response_model=SystemStatusResponse, tags=["General"])
async def get_system_status():
    """Returns current system configuration and API readiness."""
    return SystemStatusResponse(
        status="healthy",
        openai_configured=rag_engine.is_configured(),
        chroma_db_path=CHROMA_DB_PATH,
        model_name=OPENAI_MODEL
    )


@app.post("/api/resume/upload", response_model=ResumeUploadResponse, tags=["Resume"])
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    """
    Uploads a resume file (PDF, DOCX, or TXT), extracts text, chunks it,
    and indexes the embeddings inside ChromaDB.
    """
    logger.info(f"Received resume upload: {file.filename} for session: {session_id}")

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {str(e)}"
        )

    # Extract text
    try:
        text = ResumeParser.extract_text_from_bytes(content, file.filename)
    except Exception as e:
        logger.error(f"Error parsing file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse resume: {str(e)}"
        )

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded resume appeared empty or could not be decoded."
        )

    # Chunk text
    chunks = ResumeParser.chunk_resume(text, file.filename)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to generate chunks from the resume text."
        )

    # Store in ChromaDB
    try:
        stored_count = vector_store.store_resume_chunks(session_id, chunks)
    except Exception as e:
        logger.error(f"ChromaDB storage error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index resume into vector database: {str(e)}"
        )

    return ResumeUploadResponse(
        session_id=session_id,
        filename=file.filename,
        total_characters=len(text),
        chunks_stored=stored_count,
        message=f"Resume '{file.filename}' parsed and stored in ChromaDB ({stored_count} chunks indexed)."
    )


@app.post("/api/evaluate", response_model=EvaluateResponse, tags=["RAG Evaluation"])
async def evaluate_job_suitability(request: EvaluateRequest):
    """
    Evaluates whether the candidate's indexed resume matches the job description.
    Retrieves relevant resume chunks and performs deep RAG scoring.
    """
    if not request.job_description.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description cannot be empty."
        )

    # Cache JD for this session
    session_jds[request.session_id] = request.job_description

    try:
        result = rag_engine.evaluate_fit(
            session_id=request.session_id,
            job_description=request.job_description
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.post("/api/chat", response_model=ChatResponse, tags=["Conversational RAG"])
async def chat_with_bot(request: ChatRequest):
    """
    Interactive Q&A using RAG. Retrieves resume chunks grounded in the job context
    and answers candidate questions.
    """
    # Use supplied JD or fall back to cached session JD
    jd = request.job_description or session_jds.get(request.session_id, "")

    try:
        result = rag_engine.chat(
            session_id=request.session_id,
            message=request.message,
            job_description=jd,
            history=request.history
        )
        return ChatResponse(
            session_id=request.session_id,
            reply=result["reply"],
            retrieved_sources=result["retrieved_sources"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat response failed: {str(e)}"
        )


@app.post("/api/chat/stream", tags=["Conversational RAG"])
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streams LLM tokens chunk-by-chunk for near-instant responses.
    """
    jd = request.job_description or session_jds.get(request.session_id, "")

    def token_generator():
        try:
            for token in rag_engine.chat_stream(
                session_id=request.session_id,
                message=request.message,
                job_description=jd,
                history=request.history
            ):
                yield token
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.get("/api/session/{session_id}/chunks", tags=["Resume"])
async def get_session_chunks(session_id: str):
    """Fetches all stored chunks for a specific session."""
    chunks = vector_store.get_all_chunks(session_id)
    return {
        "session_id": session_id,
        "count": len(chunks),
        "chunks": chunks
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
