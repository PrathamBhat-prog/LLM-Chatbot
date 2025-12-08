from typing import Tuple, List, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils.logging_config import get_logger
from src.pipeline.retrieval import load_index
from src.core.rag_engine import load_chunks, answer_query

logger = get_logger(__name__)

app = FastAPI(
    title="Vortexus HyperRetail RAG API",
    version="0.1.0",
    description="Simple customer support RAG backend over FAQ using TF-IDF.",
)

# Globals to hold index + chunks in memory
INDEX_TUPLE: Tuple = None
CHUNKS: List[Dict] = []


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event() -> None:
    """
    Load embeddings index and FAQ chunks into memory once,
    when the API server starts.
    """
    global INDEX_TUPLE, CHUNKS

    logger.info("API startup: loading TF-IDF index and chunks...")
    INDEX_TUPLE = load_index(prefix="faq_tfidf")
    CHUNKS = load_chunks()
    logger.info("API startup: index and chunks loaded.")


@app.get("/health")
def health_check() -> dict:
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "index_loaded": INDEX_TUPLE is not None, "chunks": len(CHUNKS)}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    Takes a 'question' and returns an answer from the FAQ RAG engine.
    """
    if INDEX_TUPLE is None or not CHUNKS:
        logger.error("Index or chunks not loaded.")
        return ChatResponse(
            answer="The backend is not fully initialized yet. Please try again in a moment."
        )

    question = payload.question.strip()
    logger.info(f"/chat called with question: {question!r}")

    result = answer_query(question, INDEX_TUPLE, CHUNKS, top_k=5)
    answer = result["answer"]

    return ChatResponse(answer=answer)
