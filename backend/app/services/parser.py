import io
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document

class ResumeParser:
    """Parses resumes from PDF, DOCX, or plain text and chunks them with metadata."""

    SECTION_HEADERS = [
        "contact", "summary", "profile", "objective",
        "skills", "technical skills", "core competencies",
        "experience", "work experience", "employment history", "professional experience",
        "education", "academic background",
        "projects", "key projects", "notable projects",
        "certifications", "licenses", "publications", "awards", "honors"
    ]

    @classmethod
    def extract_text_from_bytes(cls, content: bytes, filename: str) -> str:
        """Extract clean plain text from uploaded file bytes based on extension."""
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            return cls._extract_from_pdf(content)
        elif ext in ["docx", "doc"]:
            return cls._extract_from_docx(content)
        elif ext in ["txt", "md"]:
            return content.decode("utf-8", errors="ignore")
        else:
            # Fallback attempt decoding as text
            try:
                return content.decode("utf-8")
            except Exception:
                raise ValueError(f"Unsupported file format: .{ext}. Please upload a PDF, DOCX, or TXT file.")

    @classmethod
    def _extract_from_pdf(cls, content: bytes) -> str:
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        text_parts = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text.strip())
        return "\n\n".join(text_parts)

    @classmethod
    def _extract_from_docx(cls, content: bytes) -> str:
        docx_file = io.BytesIO(content)
        doc = Document(docx_file)
        text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_parts.append(row_text)
        return "\n".join(text_parts)

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Normalizes whitespace and removes unwanted control characters."""
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @classmethod
    def chunk_resume(cls, text: str, filename: str, chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Splits resume into chunks while tagging identifiable sections where possible.
        Returns a list of dicts with 'content' and 'metadata'.
        """
        cleaned = cls.clean_text(text)
        if not cleaned:
            return []

        # Split into paragraphs/blocks first
        paragraphs = cleaned.split("\n\n")
        chunks: List[Dict[str, Any]] = []
        current_section = "General"
        current_chunk = ""

        def detect_header(line: str) -> str:
            first_line = line.strip().lower()
            # Clean punctuation
            first_line = re.sub(r'[:\-_*#]', '', first_line).strip()
            for header in cls.SECTION_HEADERS:
                if first_line == header or first_line.startswith(header):
                    return header.title()
            return ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            header_candidate = detect_header(para.split("\n")[0])
            if header_candidate:
                current_section = header_candidate

            # If adding this paragraph exceeds chunk_size and we already have content, finalize current_chunk
            if len(current_chunk) + len(para) > chunk_size and len(current_chunk) > 150:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        "section": current_section,
                        "source": filename,
                        "char_length": len(current_chunk.strip())
                    }
                })
                # Keep overlap if possible
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + "\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    "section": current_section,
                    "source": filename,
                    "char_length": len(current_chunk.strip())
                }
            })

        # Add index numbers
        for idx, c in enumerate(chunks):
            c["metadata"]["chunk_index"] = idx

        return chunks
