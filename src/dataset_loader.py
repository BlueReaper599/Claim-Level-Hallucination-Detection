import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_jsonl(filename):
    """Load a JSONL file into a Python list."""
    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def inspect_dataset(filename, n=3):
    """Print a few examples from a dataset."""
    data = load_jsonl(filename)

    print(f"Dataset: {filename}")
    print(f"Number of records: {len(data)}")
    print("\n" + "=" * 80)

    for i, record in enumerate(data[:n]):
        print(f"\nExample {i + 1}")
        print("-" * 80)

        print(f"ID: {record.get('id')}")
        print(f"Claim: {record.get('claim')}")
        print(f"Label: {record.get('label')}")

        evidence = record.get("evidence")

        if evidence:
            print(f"Evidence groups: {len(evidence)}")
            print(f"First evidence group: {evidence[0]}")
        else:
            print("Evidence: None")


if __name__ == "__main__":
    data = load_jsonl("train.jsonl")

    print(f"Total records: {len(data)}")

    label_counts = {}

    for record in data:
        label = record.get("label", "UNKNOWN")
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(data)) * 100
        print(f"{label:20s}: {count:6d} ({percentage:6.2f}%)")