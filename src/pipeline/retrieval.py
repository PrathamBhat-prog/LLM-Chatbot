from pathlib import Path
from typing import List, Dict, Tuple

import json
import numpy as np
import scipy.sparse as sp
import joblib


EMBEDDINGS_DIR = Path("data/embeddings")


def load_index(prefix: str = "faq_tfidf") -> Tuple[sp.csr_matrix, object, List[Dict]]:
    """
    Load:
      - TF-IDF embeddings matrix (sparse)
      - Fitted vectorizer
      - Metadata list (per row)
    """
    emb_path = EMBEDDINGS_DIR / f"{prefix}_embeddings.npz"
    vec_path = EMBEDDINGS_DIR / f"{prefix}_vectorizer.joblib"
    meta_path = EMBEDDINGS_DIR / f"{prefix}_metadata.jsonl"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found at: {emb_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"Vectorizer not found at: {vec_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at: {meta_path}")

    X = sp.load_npz(emb_path)
    vectorizer = joblib.load(vec_path)

    metadata: List[Dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))

    return X, vectorizer, metadata


def search(
    query: str,
    X: sp.csr_matrix,
    vectorizer,
    metadata: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Simple similarity search using cosine similarity on TF-IDF vectors.
    Returns top_k results with score + metadata.
    """
    # Embed query with same vectorizer
    q_vec = vectorizer.transform([query])  # shape: (1, num_features)

    # Cosine similarity for L2-normalized TF-IDF is just dot product
    scores = (X @ q_vec.T).toarray().ravel()  # shape: (num_docs,)

    # Get indices of top_k scores
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[Dict] = []
    for idx in top_indices:
        meta = metadata[idx]
        score = float(scores[idx])
        results.append(
            {
                "score": score,
                "question": meta.get("question"),
                "source": meta.get("source"),
                "row_index": meta.get("row_index"),
                "qa_index": meta.get("qa_index"),
                "chunk_index": meta.get("chunk_index"),
            }
        )
    return results


def main() -> None:
    print("Loading TF-IDF index...")
    X, vectorizer, metadata = load_index(prefix="faq_tfidf")
    print(f"Loaded embeddings with shape: {X.shape}")
    print(f"Loaded metadata entries: {len(metadata)}")

    print("\nEnter a query to search the FAQ (or just press Enter to exit).\n")

    while True:
        query = input("Query: ").strip()
        if not query:
            print("Exiting.")
            break

        results = search(query, X, vectorizer, metadata, top_k=5)

        print("\nTop matches:\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. [score={r['score']:.4f}] Q: {r['question']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
