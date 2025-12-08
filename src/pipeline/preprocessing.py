from pathlib import Path


RAW_DATA_DIR = Path("data/raw")


def load_raw_faq() -> str:
    """
    Load the raw FAQ markdown file as a single string.
    """
    faq_path = RAW_DATA_DIR / "faq_general.md"

    if not faq_path.exists():
        raise FileNotFoundError(f"FAQ file not found at: {faq_path}")

    text = faq_path.read_text(encoding="utf-8")
    return text


def main() -> None:
    text = load_raw_faq()
    # For now just show how many characters and first few lines
    print(f"Loaded FAQ file with {len(text)} characters.\n")
    print("Preview (first 400 chars):\n")
    print(text[:400])


if __name__ == "__main__":
    main()
