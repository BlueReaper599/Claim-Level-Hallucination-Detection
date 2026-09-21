from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}


def load_jsonl(filename):
    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def clean_record(record):
    """
    Convert one raw FEVER record into a simpler structure.
    """

    return {
        "id": record["id"],
        "claim": record["claim"],
        "label": record["label"],
        "label_id": LABEL_MAP[record["label"]],
        "evidence": record.get("evidence", []),
    }


def preprocess_dataset(filename):
    raw_records = load_jsonl(filename)

    processed_records = [
        clean_record(record)
        for record in raw_records
    ]

    return processed_records


if __name__ == "__main__":
    data = preprocess_dataset("train.jsonl")

    print(f"Processed records: {len(data)}")

    print("\nFirst processed record:")
    print(data[0])