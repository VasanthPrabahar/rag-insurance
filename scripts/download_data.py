"""Manifest-driven downloader for the auto-insurance RAG corpus.

Downloads each document in rag_insurance.manifest.DOCUMENTS to data/raw/,
skipping files that already exist. Run with:
    uv run python scripts/download_data.py [--min-success N]

By default any failure is fatal (exit 1). --min-success N tolerates
unreachable sources as long as at least N documents are present — used in
CI, where state DOI sites and the Wayback Machine sometimes block
datacenter IPs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

from rag_insurance.manifest import DOCUMENTS, DocumentMeta

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

USER_AGENT = "Mozilla/5.0 (compatible; rag-insurance-corpus-downloader/0.1)"


def download(client: httpx.Client, doc: DocumentMeta) -> bool:
    dest = DATA_DIR / doc.filename
    if dest.exists():
        print(f"SKIP    {doc.filename} (already exists)")
        return True

    try:
        response = client.get(doc.url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"FAIL    {doc.filename} <- {doc.url} ({exc})")
        return False

    dest.write_bytes(response.content)
    print(f"OK      {doc.filename} ({len(response.content):,} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-success",
        type=int,
        default=None,
        help="Tolerate failures if at least this many documents are present",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for doc in DOCUMENTS:
            if download(client, doc):
                successes += 1
            else:
                failures += 1

    print(f"\n{successes} succeeded, {failures} failed, {len(DOCUMENTS)} total")
    if args.min_success is not None:
        if successes >= args.min_success:
            if failures:
                print(f"WARN    continuing with {successes} docs (min {args.min_success})")
            return 0
        print(f"ERROR   only {successes} docs, below minimum {args.min_success}")
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
