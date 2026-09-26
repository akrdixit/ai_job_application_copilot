import os
import re
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from backend.app.config import CHROMA_DB_PATH, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

class VectorStoreManager:
    """Manages ChromaDB collections for candidate resume chunks and semantic search."""

    def __init__(self, persist_directory: str = CHROMA_DB_PATH, api_key: Optional[str] = None):
        self.persist_directory = persist_directory
        self.api_key = api_key or OPENAI_API_KEY
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.embedding_fn = self._get_embedding_function()

    def _get_embedding_function(self):
        """Initializes OpenAI embedding function or falls back to Chroma's default embedding."""
        if self.api_key and self.api_key.startswith("sk-"):
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name=OPENAI_EMBEDDING_MODEL
            )
        # Default local fallback if no valid key is provided yet
        return embedding_functions.DefaultEmbeddingFunction()

    def _format_collection_name(self, session_id: str) -> str:
        """Sanitizes session_id into a valid Chroma collection name (3-63 chars, alphanumeric + _ -)."""
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', session_id)
        if len(safe_name) < 3:
            safe_name = f"sess_{safe_name}"
        return f"resume_{safe_name}"[:63]

    def store_resume_chunks(self, session_id: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Stores chunks in a collection keyed by session_id.
        Overwrites any previous resume for this session.
        """
        col_name = self._format_collection_name(session_id)

        # Delete existing collection if present for a clean slate
        try:
            self.client.delete_collection(name=col_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=col_name,
            embedding_function=self.embedding_fn,
            metadata={"session_id": session_id}
        )

        if not chunks:
            return 0

        documents = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{session_id}_chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def search_relevant_chunks(self, session_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches vector DB for resume chunks most relevant to the query or JD requirement."""
        col_name = self._format_collection_name(session_id)
        try:
            collection = self.client.get_collection(
                name=col_name,
                embedding_function=self.embedding_fn
            )
        except Exception:
            # Collection does not exist
            return []

        count = collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [f"c_{i}" for i in range(len(docs))]

            for doc, meta, dist, chunk_id in zip(docs, metas, dists, ids):
                # Convert cosine distance to approximate similarity score (0 to 1)
                sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": round(sim_score, 4)
                })

        return formatted_results

    def search_batch_chunks(self, session_id: str, queries: List[str], top_k: int = 4) -> List[Dict[str, Any]]:
        """Batch queries ChromaDB for multiple query strings in a single call to save latency."""
        col_name = self._format_collection_name(session_id)
        try:
            collection = self.client.get_collection(
                name=col_name,
                embedding_function=self.embedding_fn
            )
        except Exception:
            return []

        count = collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = collection.query(
            query_texts=queries,
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        seen_ids = set()
        formatted_results = []
        if results and results.get("documents"):
            for doc_list, meta_list, dist_list, id_list in zip(
                results["documents"], results["metadatas"], results["distances"], results["ids"]
            ):
                for doc, meta, dist, chunk_id in zip(doc_list, meta_list, dist_list, id_list):
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                        formatted_results.append({
                            "chunk_id": chunk_id,
                            "content": doc,
                            "metadata": meta,
                            "similarity_score": round(sim_score, 4)
                        })

        return formatted_results

    def get_all_chunks(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all stored resume chunks for this session."""
        col_name = self._format_collection_name(session_id)
        try:
            collection = self.client.get_collection(
                name=col_name,
                embedding_function=self.embedding_fn
            )
        except Exception:
            return []

        data = collection.get(include=["documents", "metadatas"])
        all_chunks = []
        if data and data.get("documents"):
            for doc, meta, cid in zip(data["documents"], data["metadatas"], data["ids"]):
                all_chunks.append({
                    "chunk_id": cid,
                    "content": doc,
                    "metadata": meta
                })
        # Sort by chunk_index if present
        all_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return all_chunks

    def has_resume(self, session_id: str) -> bool:
        """Checks if a resume has been uploaded and stored for this session."""
        col_name = self._format_collection_name(session_id)
        try:
            collection = self.client.get_collection(name=col_name)
            return collection.count() > 0
        except Exception:
            return False
