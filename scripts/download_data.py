"""Manifest-driven downloader for the auto-insurance RAG corpus.

Downloads each document in DOCUMENTS to data/raw/, skipping files that
already exist. Run with: uv run python scripts/download_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

USER_AGENT = "Mozilla/5.0 (compatible; rag-insurance-corpus-downloader/0.1)"

# name: short human title
# url: direct, pre-verified link to a PDF or HTML document
# state: TX, CA, AZ, or NAIC (national)
# doc_type: "policy" | "guide" | "statute"
# filename: destination filename under data/raw/
DOCUMENTS: list[dict[str, str]] = [
    {
        "name": "TX Auto Insurance Shopping Guide",
        "url": "https://www.tdi.texas.gov/consumer/documents/auto-insurance-shopping-guide.pdf",
        "state": "TX",
        "doc_type": "guide",
        "filename": "tx-auto-insurance-shopping-guide.pdf",
    },
    {
        "name": "TX Auto Insurance Comparison Worksheet",
        "url": "https://www.tdi.texas.gov/consumer/documents/autoworksheet.pdf",
        "state": "TX",
        "doc_type": "guide",
        "filename": "tx-auto-insurance-comparison-worksheet.pdf",
    },
    {
        "name": "TX Prescribed Auto Insurance ID Card (Form PC418, 28 TAC 5.204)",
        "url": "https://www.tdi.texas.gov/forms/pcpersonal/pc418IDcard.pdf",
        "state": "TX",
        "doc_type": "policy",
        "filename": "tx-auto-id-card-form-pc418.pdf",
    },
    {
        "name": "TX Adopted Rule: Named-Driver Auto Policy Disclosure (28 TAC 5.208)",
        "url": "https://www.tdi.texas.gov/rules/2014/documents/3756.pdf",
        "state": "TX",
        "doc_type": "statute",
        "filename": "tx-named-driver-disclosure-rule-3756.pdf",
    },
    {
        "name": "CA Automobile Insurance Consumer Guide (2025)",
        "url": (
            "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/"
            "01-auto/upload/IG-Auto-Insurance-Updated-020525.pdf"
        ),
        "state": "CA",
        "doc_type": "guide",
        "filename": "ca-automobile-insurance-guide-2025.pdf",
    },
    {
        "name": "CA Shopping for Auto Insurance Guide",
        "url": (
            "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/"
            "01-auto/upload/IG-WTDTIH-Updated-092623.pdf"
        ),
        "state": "CA",
        "doc_type": "guide",
        "filename": "ca-shopping-for-auto-insurance-guide.pdf",
    },
    {
        "name": "CA Low Cost Automobile Insurance Program Pamphlet",
        "url": (
            "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/"
            "01-auto/lca/upload/Pamphlet-English-1.pdf"
        ),
        "state": "CA",
        "doc_type": "guide",
        "filename": "ca-low-cost-auto-insurance-pamphlet.pdf",
    },
    {
        # difi.az.gov is behind a Cloudflare bot wall (see NOTES/phase0.md),
        # so this is fetched via its Wayback Machine mirror instead.
        "name": "AZ Consumer Guide to How Insurers Use Credit Information",
        "url": (
            "https://web.archive.org/web/20250309205041/"
            "https://difi.az.gov/sites/default/files/"
            "Consumer%20Guide%20to%20How%20Insurers%20Use%20Credit_20180618.pdf"
        ),
        "state": "AZ",
        "doc_type": "guide",
        "filename": "az-consumer-guide-credit-scoring.pdf",
    },
    {
        "name": "AZ 2023 Auto Premium Report",
        "url": (
            "https://web.archive.org/web/20250225200727/"
            "https://difi.az.gov/sites/default/files/"
            "2023%20Arizona%20Auto%20Premium%20Report%20(2).pdf"
        ),
        "state": "AZ",
        "doc_type": "guide",
        "filename": "az-2023-auto-premium-report.pdf",
    },
    {
        "name": "NAIC A Consumer's Guide to Auto Insurance",
        "url": "https://content.naic.org/sites/default/files/publication-aut-pp-consumer-auto.pdf",
        "state": "NAIC",
        "doc_type": "guide",
        "filename": "naic-consumers-guide-auto-insurance.pdf",
    },
    {
        "name": "NAIC Consumer Shopping Tool for Auto Insurance",
        "url": (
            "https://content.naic.org/sites/default/files/inline-files/"
            "topic_transparency_readability_consumer_auto_tool.pdf"
        ),
        "state": "NAIC",
        "doc_type": "guide",
        "filename": "naic-consumer-shopping-tool-auto.pdf",
    },
    {
        "name": "NAIC A Shopping Tool for Auto Insurance",
        "url": "https://content.naic.org/sites/default/files/consumer-auto-shopping-tool.pdf",
        "state": "NAIC",
        "doc_type": "guide",
        "filename": "naic-shopping-tool-auto-insurance.pdf",
    },
]


def download(client: httpx.Client, doc: dict[str, str]) -> bool:
    dest = DATA_DIR / doc["filename"]
    if dest.exists():
        print(f"SKIP    {doc['filename']} (already exists)")
        return True

    try:
        response = client.get(doc["url"], follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"FAIL    {doc['filename']} <- {doc['url']} ({exc})")
        return False

    dest.write_bytes(response.content)
    print(f"OK      {doc['filename']} ({len(response.content):,} bytes)")
    return True


def main() -> int:
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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
