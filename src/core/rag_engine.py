import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Ensure project root is on sys.path so "src.*" imports work reliably
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.logging_config import get_logger
from src.pipeline.retrieval import load_index, search

logger = get_logger(__name__)
PROCESSED_DATA_DIR = Path("data/processed")


def load_chunks() -> List[Dict]:
    """
    Load processed FAQ chunks from JSONL.
    This must match the same order used for building embeddings/metadata.
    """
    chunks_path = PROCESSED_DATA_DIR / "faq_chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

    chunks: List[Dict] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def answer_query(
    query: str,
    index_tuple: Tuple,
    chunks: List[Dict],
    top_k: int = 5,
) -> Dict:
    """
    Run retrieval and return an answer directly from the best FAQ chunk.
    Only the answer text is returned (no FAQ question shown to the user).
    """
    X, vectorizer, metadata = index_tuple

    logger.info(f"Running retrieval for query: {query!r}")
    results = search(query, X, vectorizer, metadata, top_k=top_k)

    if not results:
        logger.warning("No retrieval results found.")
        return {
            "answer": (
                "I’m not able to find any information related to your question "
                "in the current FAQ knowledge base."
            ),
            "results": [],
        }

    # Take the single best match
    best = results[0]
    row_index = best["row_index"]
    best_chunk = chunks[row_index]

    answer_chunk = (best_chunk.get("answer_chunk") or "").strip()

    if not answer_chunk:
        answer_chunk = (
            "I don’t have a clear answer stored for this question in the FAQ."
        )

    # Final answer: just the answer text, no internal FAQ question
    final_answer = answer_chunk

    return {
        "answer": final_answer,
        "results": results,
    }


def main() -> None:
    logger.info("Loading TF-IDF index and chunks...")
    index_tuple = load_index(prefix="faq_tfidf")
    chunks = load_chunks()
    logger.info("Index and chunks loaded successfully.")

    print("\nVortexus HyperRetail – RAG Demo (direct FAQ answers)")
    print("Ask a customer support question (empty input to exit).\n")

    while True:
        query = input("You: ").strip()
        if not query:
            print("Exiting.")
            break

        result = answer_query(query, index_tuple, chunks, top_k=5)
        answer = result["answer"]

        print("\nAssistant:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
