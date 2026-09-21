import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "shared_task_dev.jsonl"
)


def load_jsonl(filename):

    records = []

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                records.append(
                    json.loads(line)
                )

    return records


def get_gold_sentences(record):

    gold = set()

    for evidence_group in record.get(
        "evidence",
        []
    ):

        for evidence_item in evidence_group:

            if len(evidence_item) >= 4:

                page_id = evidence_item[2]

                sentence_id = evidence_item[3]

                if (
                    page_id is not None
                    and sentence_id is not None
                ):

                    gold.add(
                        (
                            str(page_id),
                            str(sentence_id)
                        )
                    )

    return gold


def main():

    data = load_jsonl(
        DEV_FILE
    )

    print("=" * 80)
    print("INSPECTING MISSING SENTENCE-LEVEL GOLD EVIDENCE")
    print("=" * 80)

    missing = 0

    for i, record in enumerate(
        data[:100]
    ):

        gold = get_gold_sentences(
            record
        )

        if not gold:

            missing += 1

            print(
                f"\nExample index: {i}"
            )

            print(
                f"ID: {record.get('id')}"
            )

            print(
                f"Claim: {record.get('claim')}"
            )

            print(
                f"Label: {record.get('label')}"
            )

            print(
                "Raw evidence:"
            )

            print(
                json.dumps(
                    record.get("evidence"),
                    indent=2
                )
            )

            print("-" * 80)

    print(
        f"\nMissing/empty sentence-level "
        f"evidence: {missing}"
    )


if __name__ == "__main__":
    main()