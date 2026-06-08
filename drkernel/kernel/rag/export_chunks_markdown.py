import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z0-9_:+.-]+")
MOJIBAKE_MARKERS = ("â", "Â", "Ã", "â", "ð")


def count_tokens(text: str) -> int:
    lexical_tokens = len(TOKEN_RE.findall(text))
    whitespace_tokens = len(text.split())
    return max(lexical_tokens, whitespace_tokens)


def fix_mojibake(text: str) -> str:
    if not text or not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def chunk_sort_key(chunk: dict):
    chunk_id = chunk["id"]
    if "::chunk::" in chunk_id:
        prefix, suffix = chunk_id.rsplit("::chunk::", 1)
        try:
            return prefix, int(suffix)
        except ValueError:
            return prefix, suffix
    return chunk_id, 0


def safe_filename(name: str) -> str:
    filename = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return filename or "untitled"


def load_chunks(chunks_path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            grouped[chunk["source_id"]].append(chunk)

    for source_id in grouped:
        grouped[source_id].sort(key=chunk_sort_key)
    return dict(grouped)


def render_source_markdown(source_id: str, chunks: list[dict]) -> str:
    first = chunks[0]
    title = fix_mojibake(first.get("title") or source_id)
    lines = [
        f"# {title}",
        "",
        f"- `source_id`: `{source_id}`",
        f"- `url`: {first.get('url', '')}",
        f"- `category`: `{first.get('category', '')}`",
        f"- `num_chunks`: `{len(chunks)}`",
    ]
    tags = first.get("tags") or []
    if tags:
        lines.append(f"- `tags`: `{', '.join(tags)}`")
    lines.append("")

    for idx, chunk in enumerate(chunks):
        chunk_text = fix_mojibake(chunk["text"])
        token_estimate = count_tokens(chunk_text)
        word_count = len(chunk_text.split())
        lines.extend(
            [
                f"## Chunk {idx}",
                "",
                f"- `chunk_id`: `{chunk['id']}`",
                f"- `token_estimate`: `{token_estimate}`",
                f"- `word_count`: `{word_count}`",
                "",
                "~~~text",
                chunk_text.rstrip(),
                "~~~",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def render_index(source_rows: list[dict]) -> str:
    lines = [
        "# Chunk Review",
        "",
        "This directory contains the exported RAG chunks in Markdown for manual review.",
        "",
        "| Source | Chunks | File |",
        "| --- | ---: | --- |",
    ]

    for row in source_rows:
        lines.append(
            f"| `{row['source_id']}` | {row['num_chunks']} | [{row['title']}]({row['filename']}) |"
        )

    lines.append("")
    return "\n".join(lines)


def export_chunks(chunks_path: Path, output_dir: Path) -> Path:
    grouped = load_chunks(chunks_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = []
    for source_id, chunks in sorted(grouped.items()):
        first = chunks[0]
        filename = safe_filename(source_id) + ".md"
        file_path = output_dir / filename
        file_path.write_text(render_source_markdown(source_id, chunks), encoding="utf-8")
        source_rows.append(
            {
                "source_id": source_id,
                "title": fix_mojibake(first.get("title") or source_id),
                "num_chunks": len(chunks),
                "filename": filename,
            }
        )

    index_path = output_dir / "README.md"
    index_path.write_text(render_index(source_rows), encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chunked manual RAG data to Markdown for review.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path(__file__).resolve().parent / "processed" / "chunks.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "review_md",
    )
    args = parser.parse_args()

    index_path = export_chunks(args.chunks_path, args.output_dir)
    print(index_path)


if __name__ == "__main__":
    main()
