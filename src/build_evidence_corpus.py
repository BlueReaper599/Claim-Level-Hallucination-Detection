import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WIKI_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "wiki_pages"
    / "wiki-pages"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "evidence_corpus.jsonl"
)


def parse_lines(page):
    """
    Convert the FEVER 'lines' field into sentence records.

    Each line has the form:

        sentence_id<TAB>sentence_text<TAB>...</...>
    """

    records = []

    lines = page.get("lines", "")

    if not lines:
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
            "page_id": page["id"],
            "sentence_id": sentence_id,
            "text": sentence_text,
        })

    return records


def build_corpus():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    wiki_files = sorted(WIKI_DIR.glob("*.jsonl"))

    print(f"Found {len(wiki_files)} Wikipedia files.")

    total_pages = 0
    total_sentences = 0

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

                    if not page.get("id"):
                        continue

                    total_pages += 1

                    sentences = parse_lines(page)

                    for sentence in sentences:
                        out.write(
                            json.dumps(
                                sentence,
                                ensure_ascii=False
                            )
                            + "\n"
                        )

                        total_sentences += 1

    print("\n" + "=" * 60)
    print("Evidence corpus built successfully!")
    print("=" * 60)

    print(f"Pages processed     : {total_pages:,}")
    print(f"Sentences extracted : {total_sentences:,}")
    print(f"Output file         : {OUTPUT_FILE}")


if __name__ == "__main__":
    build_corpus()