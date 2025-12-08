from pathlib import Path
import re
import json
from typing import List, Dict


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def extract_qa_pairs() -> List[Dict[str, str]]:
    """
    Extract Q–A pairs from the markdown FAQ.
    Returns a list of dicts like:
    [{'question': '...', 'answer': '...'}, ...]
    """
    faq_path = RAW_DATA_DIR / "faq_general.md"
    text = faq_path.read_text(encoding="utf-8")

    # Pattern:
    # ### Q12. Some question text
    # A: Some answer text (possibly multi-line) ... until next ### or EOF
    pattern = r"###\s*Q\d+\.\s*(.*?)\nA:\s*(.*?)(?=\n###|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    qa_pairs: List[Dict[str, str]] = []
    for q, a in matches:
        qa_pairs.append(
            {
                "question": q.strip(),
                # normalize newlines inside answer
                "answer": a.strip().replace("\n", " "),
            }
        )

    return qa_pairs


def split_text_into_chunks(text: str, max_chars: int = 450) -> List[str]:
    """
    Split a long answer into smaller chunks based on sentence boundaries,
    without exceeding max_chars per chunk (rough heuristic).
    """
    # Simple sentence split based on punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # If adding this sentence would exceed the limit, start a new chunk.
        if len(current) + len(sent) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            if current:
                current += " " + sent
            else:
                current = sent

    if current:
        chunks.append(current.strip())

    return chunks


def build_chunks_from_qa(qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Turn QA pairs into chunk dictionaries.
    Each long answer may become multiple chunks.
    """
    chunks: List[Dict[str, str]] = []

    for qa_idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        answer = qa["answer"]

        answer_chunks = split_text_into_chunks(answer, max_chars=450)

        for chunk_idx, chunk_text in enumerate(answer_chunks):
            chunk_id = f"faq_{qa_idx}_chunk_{chunk_idx}"

            chunks.append(
                {
                    "id": chunk_id,
                    "question": question,
                    "answer_chunk": chunk_text,
                    "qa_index": qa_idx,
                    "chunk_index": chunk_idx,
                    "source": "faq_general.md",
                }
            )

    return chunks


def save_chunks_to_jsonl(chunks: List[Dict[str, str]], output_path: Path) -> None:
    """
    Save chunks as JSONL so each line is a JSON object.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    qa_pairs = extract_qa_pairs()
    print(f"Extracted {len(qa_pairs)} QA pairs.")

    chunks = build_chunks_from_qa(qa_pairs)
    print(f"Generated {len(chunks)} chunks from QA pairs.\n")

    # Preview first few chunks
    for c in chunks[:3]:
        print(f"ID: {c['id']}")
        print(f"Q: {c['question']}")
        print(f"Chunk: {c['answer_chunk'][:120]}...")
        print("-" * 60)

    # Save to processed file
    output_file = PROCESSED_DATA_DIR / "faq_chunks.jsonl"
    save_chunks_to_jsonl(chunks, output_file)
    print(f"\nSaved chunks to: {output_file}")


if __name__ == "__main__":
    main()
