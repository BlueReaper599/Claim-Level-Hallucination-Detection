import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
WIKI_DIR = RAW_DATA_DIR / "wiki_pages" / "wiki-pages"

DEV_FILE = RAW_DATA_DIR / "shared_task_dev.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dev_evidence_corpus.jsonl"


def load_dev_pages():
    """Collect Wikipedia page IDs referenced by FEVER dev evidence."""

    page_ids = set()

    with open(DEV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            for evidence_group in record.get("evidence", []):
                for evidence_item in evidence_group:
                    if len(evidence_item) >= 3:
                        page_ids.add(evidence_item[2])

    return page_ids


def parse_page(page):
    """Extract sentence-level evidence from a Wikipedia page."""

    records = []

    page_id = page.get("id")
    lines = page.get("lines", "")

    if not page_id or not lines:
        return records

    for line in lines.split("\n"):
        if not line.strip():
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        try:
            sentence_id = int(parts[0])
        except ValueError:
            continue

        sentence_text = parts[1].strip()

        if not sentence_text:
            continue

        records.append({
            "page_id": page_id,
            "sentence_id": sentence_id,
            "text": sentence_text,
        })

    return records


def build_corpus():
    print("Loading page IDs from FEVER development set...")

    target_pages = load_dev_pages()

    print(f"Unique evidence pages required: {len(target_pages):,}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_sentences = 0

    wiki_files = sorted(WIKI_DIR.glob("*.jsonl"))

    print(f"Searching {len(wiki_files)} Wikipedia files...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for file_number, wiki_file in enumerate(wiki_files, start=1):

            print(
                f"Processing {file_number}/{len(wiki_files)}: "
                f"{wiki_file.name}"
            )

            with open(wiki_file, "r", encoding="utf-8") as f:

                for line in f:

                    if not line.strip():
                        continue

                    page = json.loads(line)

                    page_id = page.get("id")

                    if page_id not in target_pages:
                        continue

                    total_pages += 1

                    sentences = parse_page(page)

                    for sentence in sentences:

                        out.write(
                            json.dumps(
                                sentence,
                                ensure_ascii=False
                            ) + "\n"
                        )

                        total_sentences += 1

    print("\n" + "=" * 60)
    print("Development evidence corpus built!")
    print("=" * 60)

    print(f"Target pages        : {len(target_pages):,}")
    print(f"Pages found         : {total_pages:,}")
    print(f"Sentences extracted : {total_sentences:,}")
    print(f"Output file         : {OUTPUT_FILE}")


if __name__ == "__main__":
    build_corpus()