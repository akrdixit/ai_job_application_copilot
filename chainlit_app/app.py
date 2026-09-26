import os
import uuid
import httpx
import chainlit as cl

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def is_likely_job_description(text: str) -> bool:
    """Detects if a user message is a Job Description rather than a general query or greeting."""
    cleaned = text.strip().lower()
    if len(cleaned) < 100:
        return False
    jd_indicators = [
        "requirements", "responsibilities", "qualifications", "job description",
        "about the role", "about the job", "we are looking for", "skills required",
        "preferred qualifications", "minimum qualifications", "what you'll do",
        "what you will do", "experience required", "key responsibilities", "job title"
    ]
    if any(ind in cleaned for ind in jd_indicators):
        return True
    # Long text with line breaks typically denotes a pasted job description
    if len(cleaned) > 250 and ("\n" in text):
        return True
    return False


@cl.on_chat_start
async def on_start():
    session_id = f"user_{uuid.uuid4().hex[:8]}"
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("job_description", "")
    cl.user_session.set("resume_uploaded", False)

    await cl.Message(content="# AI Job Application Chatbot!").send()


async def handle_resume_upload(uploaded_file, announce_next_step: bool = True) -> bool:
    """Indexes the resume. Returns True on success."""
    session_id = cl.user_session.get("session_id")

    file_bytes = None
    if hasattr(uploaded_file, "path") and uploaded_file.path and os.path.exists(uploaded_file.path):
        with open(uploaded_file.path, "rb") as f:
            file_bytes = f.read()
    elif hasattr(uploaded_file, "content") and uploaded_file.content:
        file_bytes = uploaded_file.content
    else:
        await cl.Message(content="❌ Could not read uploaded file content.").send()
        return False

    async with cl.Step(name="Indexing Resume into ChromaDB Vector Store") as step:
        step.output = f"Extracting sections & embedding chunks for '{uploaded_file.name}'..."
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                files_payload = {
                    "file": (uploaded_file.name, file_bytes, "application/octet-stream")
                }
                data_payload = {"session_id": session_id}
                
                resp = await client.post(
                    f"{BACKEND_URL}/api/resume/upload",
                    files=files_payload,
                    data=data_payload
                )

                if resp.status_code != 200:
                    error_detail = resp.json().get("detail", resp.text)
                    step.output = f"❌ Error: {error_detail}"
                    await cl.Message(content=f"⚠️ Failed to index resume: {error_detail}").send()
                    return False

                res_json = resp.json()
                step.output = f"✅ Successfully indexed {res_json.get('chunks_stored', 0)} chunks into ChromaDB!"
                cl.user_session.set("resume_uploaded", True)
                cl.user_session.set("resume_name", uploaded_file.name)
        except Exception as e:
            step.output = f"❌ Connection Error: {str(e)}"
            await cl.Message(
                content=f"⚠️ Could not connect to FastAPI backend at `{BACKEND_URL}`. Make sure the backend server is running!"
            ).send()
            return False

    content = f"✅ Resume **{uploaded_file.name}** successfully indexed in ChromaDB!"
    if announce_next_step:
        content += "\n\n📋 **Next Step:** Paste your target **Job Description** below to analyze your match and get your suitability score."
    await cl.Message(content=content).send()
    return True


async def evaluate_and_display(session_id: str, job_description: str):
    """Triggers RAG suitability analysis and renders structured scorecard."""
    async with cl.Step(name="Analyzing Job Match with RAG") as step:
        step.output = "Querying ChromaDB for relevant resume sections and running LLM evaluation..."
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{BACKEND_URL}/api/evaluate",
                    json={"session_id": session_id, "job_description": job_description}
                )

                if resp.status_code != 200:
                    err = resp.json().get("detail", resp.text)
                    step.output = f"❌ Evaluation failed: {err}"
                    await cl.Message(content=f"⚠️ Evaluation failed: {err}").send()
                    return

                data = resp.json()
                suitability = data["suitability"]
                step.output = f"✅ Evaluation Complete! Overall Match: {suitability['overall_score']}%"
        except Exception as e:
            step.output = f"❌ Error: {str(e)}"
            await cl.Message(content=f"⚠️ Failed to evaluate: {str(e)}").send()
            return

    # Format Score & Badges
    score = suitability["overall_score"]
    fit_level = suitability["fit_level"]
    badge = "🟢" if score >= 80 else ("🟡" if score >= 60 else "🔴")

    strengths_md = "\n".join([f"- ✅ {s}" for s in suitability.get("key_strengths", [])])
    gaps_md = "\n".join([f"- ⚠️ {g}" for g in suitability.get("missing_skills_and_gaps", [])])
    keywords_md = ", ".join([f"`{k}`" for k in suitability.get("critical_missing_keywords", [])]) or "None identified"
    recommendations_md = "\n".join([f"- 💡 {r}" for r in suitability.get("actionable_recommendations", [])])

    scorecard_md = f"""
## {badge} Job Suitability Scorecard: **{score}% ({fit_level})**

> **Executive Summary:**  
> {suitability['summary']}

---

### 📊 Score Breakdown
| Metric | Score | Rating |
| :--- | :---: | :--- |
| **Technical & Skills Match** | **{suitability['skills_match_score']}%** | {'High' if suitability['skills_match_score'] >= 75 else 'Moderate' if suitability['skills_match_score'] >= 50 else 'Low'} |
| **Experience & Seniority** | **{suitability['experience_match_score']}%** | {'High' if suitability['experience_match_score'] >= 75 else 'Moderate' if suitability['experience_match_score'] >= 50 else 'Low'} |
| **Education & Credentials** | **{suitability['education_match_score']}%** | {'High' if suitability['education_match_score'] >= 75 else 'Moderate' if suitability['education_match_score'] >= 50 else 'Low'} |

---

### 🌟 Key Strengths & Exact Matches
{strengths_md or "_No direct strengths highlighted._"}

---

### ⚠️ Missing Skills & Requirements Gaps
{gaps_md or "_No significant gaps detected!_"}

---

### 🔑 Critical Missing ATS Keywords
{keywords_md}

---

### 🎯 Actionable Tailoring Recommendations
{recommendations_md or "_Your resume already aligns well._"}

---
💬 **What would you like to do next?** You can ask questions directly, or try asking:
- *"Rewrite my most recent job bullet points to incorporate the missing keywords"*
- *"Draft a customized cover letter highlighting my key strengths for this role"*
- *"What interview questions might the hiring team ask based on my gaps?"*
"""
    await cl.Message(content=scorecard_md).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id", "default")
    job_description = cl.user_session.get("job_description", "")
    history = cl.user_session.get("chat_history", [])
    resume_uploaded = cl.user_session.get("resume_uploaded", False)

    text = (message.content or "").strip()
    text_is_jd = is_likely_job_description(text)

    # 1. Check if a file was attached in this message. Keep going afterwards so a
    # question or JD sent together with the resume isn't dropped.
    if message.elements:
        for elem in message.elements:
            if hasattr(elem, "path") or hasattr(elem, "content"):
                announce = not text_is_jd and not job_description
                resume_uploaded = await handle_resume_upload(elem, announce_next_step=announce) or resume_uploaded
                break

        if resume_uploaded and job_description and not text:
            # JD was provided before the resume: evaluate now
            await evaluate_and_display(session_id, job_description)
            return
        if not text:
            return

    # 2. A pasted Job Description (new or replacing the previous one) triggers a full evaluation
    if text_is_jd:
        job_description = text
        cl.user_session.set("job_description", job_description)
        if resume_uploaded:
            await evaluate_and_display(session_id, job_description)
        else:
            await cl.Message(
                content="📋 Got the job description! Attach your **resume** (PDF/DOCX/TXT) and I'll score how well it matches."
            ).send()
        return

    # 3. Handle greetings, scope inquiries, career questions, and ongoing RAG chat
    history.append({"role": "user", "content": message.content})

    # 3. Stream response directly for instant typing speed
    msg = cl.Message(content="")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/chat/stream",
                json={
                    "session_id": session_id,
                    "message": message.content,
                    "job_description": job_description,
                    "history": history
                }
            ) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    error_msg = f"⚠️ Request failed: {err_bytes.decode('utf-8', errors='ignore')}"
                    await msg.stream_token(error_msg)
                else:
                    async for chunk in response.aiter_text():
                        await msg.stream_token(chunk)
        await msg.update()
    except Exception as e:
        await msg.stream_token(f"⚠️ Could not reach backend: {str(e)}")
        await msg.update()
        return

    # Save assistant message to history
    history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("chat_history", history)
