import argparse
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


DEFAULT_TIMEOUT = 60
USER_AGENT = "KernelGYM-RAG/1.0"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _extract_title(html: str, fallback: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.text.strip():
        return soup.title.text.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return fallback


def download_sources(manifest_path: Path, output_dir: Path, timeout: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    records = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for source in manifest["sources"]:
        source_id = source["id"]
        url = source["url"]
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        local_name = _safe_name(source_id) + ".html"
        local_path = raw_dir / local_name
        local_path.write_text(response.text, encoding="utf-8")
        records.append(
            {
                **source,
                "local_path": str(local_path),
                "content_type": response.headers.get("content-type", ""),
                "fetched_title": _extract_title(response.text, source["title"]),
                "status_code": response.status_code,
            }
        )

    lock_path = output_dir / "downloaded_manifest.json"
    lock_path.write_text(json.dumps({"sources": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return lock_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official manuals for KernelGYM RAG.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manual_manifest.json"),
        help="Path to the manual manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to store raw downloaded manuals.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    lock_path = download_sources(args.manifest, args.output_dir, args.timeout)
    print(lock_path)


if __name__ == "__main__":
    main()
