from pathlib import Path
from typing import List, Dict
import json

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


PROCESSED_DATA_DIR = Path("data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")


def load_chunks() -> List[Dict]:
    """
    Load chunks from the processed JSONL file.
    Each line is a JSON object with:
      - id
      - question
      - answer_chunk
      - qa_index
      - chunk_index
      - source
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


def build_texts_for_embedding(chunks: List[Dict]) -> List[str]:
    """
    Build text to embed for each chunk.
    We'll combine question + answer chunk for richer context.
    """
    texts: List[str] = []
    for c in chunks:
        q = c.get("question", "")
        a = c.get("answer_chunk", "")
        combined = f"Q: {q} A: {a}"
        texts.append(combined)
    return texts


def generate_tfidf_embeddings(texts: List[str]):
    """
    Generate TF-IDF embeddings using a local scikit-learn model.
    Returns:
      - sparse matrix (num_texts, num_features)
      - fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=4096,   # cap feature space size
        ngram_range=(1, 2),  # unigrams + bigrams
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def save_embeddings(
    X, chunks: List[Dict], vectorizer: TfidfVectorizer, prefix: str = "faq_tfidf"
) -> None:
    """
    Save embeddings matrix, vectorizer, and metadata.
    - embeddings: data/embeddings/<prefix>_embeddings.npz
    - vectorizer: data/embeddings/<prefix>_vectorizer.joblib
    - metadata:   data/embeddings/<prefix>_metadata.jsonl
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = EMBEDDINGS_DIR / f"{prefix}_embeddings.npz"
    vec_path = EMBEDDINGS_DIR / f"{prefix}_vectorizer.joblib"
    meta_path = EMBEDDINGS_DIR / f"{prefix}_metadata.jsonl"

    # Save sparse matrix
    sp.save_npz(emb_path, X)

    # Save vectorizer
    joblib.dump(vectorizer, vec_path)

    # Save metadata
    with meta_path.open("w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            meta = {
                "row_index": idx,
                "id": chunk.get("id"),
                "question": chunk.get("question"),
                "qa_index": chunk.get("qa_index"),
                "chunk_index": chunk.get("chunk_index"),
                "source": chunk.get("source"),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved vectorizer to:  {vec_path}")
    print(f"Saved metadata to:    {meta_path}")


def main() -> None:
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print("Building texts for embedding...")
    texts = build_texts_for_embedding(chunks)
    print(f"Prepared {len(texts)} texts.")

    print("Generating TF-IDF embeddings (local, CPU only)...")
    X, vectorizer = generate_tfidf_embeddings(texts)
    print(f"Embeddings shape: {X.shape}")

    print("Saving embeddings, vectorizer, and metadata...")
    save_embeddings(X, chunks, vectorizer, prefix="faq_tfidf")

    print("Done.")


if __name__ == "__main__":
    main()
