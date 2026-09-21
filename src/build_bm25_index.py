import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dev_evidence_corpus.jsonl"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bm25_index"
)

BM25_FILE = INDEX_DIR / "bm25.pkl"
METADATA_FILE = INDEX_DIR / "metadata.jsonl"


def tokenize(text):
    """
    Simple BM25 tokenizer.

    Converts text to lowercase and keeps
    alphanumeric word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def load_corpus():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def build_index():

    print("=" * 70)
    print("Loading development evidence corpus")
    print("=" * 70)

    records = load_corpus()

    print(f"Evidence sentences: {len(records):,}")

    print("\nTokenizing evidence...")

    tokenized_corpus = [
        tokenize(record["text"])
        for record in records
    ]

    print("Tokenization complete.")

    print("\nBuilding BM25 index...")

    bm25 = BM25Okapi(tokenized_corpus)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Saved BM25 index: {BM25_FILE}")

    # Save metadata in exactly the same order as
    # the documents used to construct BM25.
    with open(METADATA_FILE, "w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Saved metadata: {METADATA_FILE}")

    print("\n" + "=" * 70)
    print("BM25 index built successfully!")
    print("=" * 70)

    print(f"Documents: {len(records):,}")


if __name__ == "__main__":
    build_index()