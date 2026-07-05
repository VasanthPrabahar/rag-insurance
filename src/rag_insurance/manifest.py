"""Corpus manifest: every document in data/raw/ and its metadata.

Single source of truth shared by scripts/download_data.py (download) and
rag_insurance.ingest.parser (metadata attachment).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DocType = Literal["policy", "guide", "statute"]


class DocumentMeta(BaseModel):
    name: str
    url: str
    state: str
    doc_type: DocType
    filename: str


DOCUMENTS: list[DocumentMeta] = [
    DocumentMeta(
        name="TX Auto Insurance Shopping Guide",
        url="https://www.tdi.texas.gov/consumer/documents/auto-insurance-shopping-guide.pdf",
        state="TX",
        doc_type="guide",
        filename="tx-auto-insurance-shopping-guide.pdf",
    ),
    DocumentMeta(
        name="TX Auto Insurance Comparison Worksheet",
        url="https://www.tdi.texas.gov/consumer/documents/autoworksheet.pdf",
        state="TX",
        doc_type="guide",
        filename="tx-auto-insurance-comparison-worksheet.pdf",
    ),
    DocumentMeta(
        name="TX Prescribed Auto Insurance ID Card (Form PC418, 28 TAC 5.204)",
        url="https://www.tdi.texas.gov/forms/pcpersonal/pc418IDcard.pdf",
        state="TX",
        doc_type="policy",
        filename="tx-auto-id-card-form-pc418.pdf",
    ),
    DocumentMeta(
        name="TX Adopted Rule: Named-Driver Auto Policy Disclosure (28 TAC 5.208)",
        url="https://www.tdi.texas.gov/rules/2014/documents/3756.pdf",
        state="TX",
        doc_type="statute",
        filename="tx-named-driver-disclosure-rule-3756.pdf",
    ),
    DocumentMeta(
        name="CA Automobile Insurance Consumer Guide (2025)",
        url=(
            "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/"
            "01-auto/upload/IG-Auto-Insurance-Updated-020525.pdf"
        ),
        state="CA",
        doc_type="guide",
        filename="ca-automobile-insurance-guide-2025.pdf",
    ),
    DocumentMeta(
        name="CA Shopping for Auto Insurance Guide",
        url=(
            "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/"
            "01-auto/upload/IG-WTDTIH-Updated-092623.pdf"
        ),
        state="CA",
        doc_type="guide",
        filename="ca-shopping-for-auto-insurance-guide.pdf",
    ),
    DocumentMeta(
        name="CA Low Cost Automobile Insurance Program Pamphlet",
        url=(
            "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/"
            "01-auto/lca/upload/Pamphlet-English-1.pdf"
        ),
        state="CA",
        doc_type="guide",
        filename="ca-low-cost-auto-insurance-pamphlet.pdf",
    ),
    # difi.az.gov is behind a Cloudflare bot wall (see NOTES/phase0.md),
    # so the AZ documents are fetched via their Wayback Machine mirrors.
    DocumentMeta(
        name="AZ Consumer Guide to How Insurers Use Credit Information",
        url=(
            "https://web.archive.org/web/20250309205041/"
            "https://difi.az.gov/sites/default/files/"
            "Consumer%20Guide%20to%20How%20Insurers%20Use%20Credit_20180618.pdf"
        ),
        state="AZ",
        doc_type="guide",
        filename="az-consumer-guide-credit-scoring.pdf",
    ),
    # Removed in Phase 2: "AZ 2023 Auto Premium Report" — 233 chunks of
    # rate-table noise that made up ~60% of the corpus (see NOTES/phase2.md).
    DocumentMeta(
        name="AZ New Driver's Guide to Auto Insurance",
        url=(
            "https://web.archive.org/web/20250613195427/"
            "https://difi.az.gov/sites/default/files/"
            "New%20Driver%27s%20Guide%20to%20Auto%20Insurance_20180618.pdf"
        ),
        state="AZ",
        doc_type="guide",
        filename="az-new-drivers-guide.pdf",
    ),
    DocumentMeta(
        name="TX Automobile Insurance Guide (cb020)",
        url="https://www.tdi.texas.gov/pubs/consumer/cb020.html",
        state="TX",
        doc_type="guide",
        filename="tx-automobile-insurance-guide-cb020.html",
    ),
    # Full ISO Personal Auto Policy specimens: complete contracts with
    # definitions, Parts A-D, exclusions, and general provisions.
    DocumentMeta(
        name="ISO Personal Auto Policy specimen PP 00 01 06 98 (Nevada DOI)",
        url=(
            "https://doi.nv.gov/uploadedfiles/doinvgov/_public-documents/"
            "Consumers/PP_00_01_06_98.pdf"
        ),
        state="NV",
        doc_type="policy",
        filename="iso-personal-auto-policy-pp-00-01-06-98.pdf",
    ),
    DocumentMeta(
        name="ISO Personal Auto Policy specimen PP 00 01 09 18 (Virginia SCC)",
        url=(
            "https://www.scc.virginia.gov/media/sccvirginiagov-home/"
            "regulated-industries/insurance/insurance-companies/"
            "property-casualty-companies/personal-commercial-auto-forms/"
            "pp-00-01-09-18.pdf"
        ),
        state="VA",
        doc_type="policy",
        filename="iso-personal-auto-policy-pp-00-01-09-18.pdf",
    ),
    DocumentMeta(
        name="NAIC A Consumer's Guide to Auto Insurance",
        url="https://content.naic.org/sites/default/files/publication-aut-pp-consumer-auto.pdf",
        state="NAIC",
        doc_type="guide",
        filename="naic-consumers-guide-auto-insurance.pdf",
    ),
    DocumentMeta(
        name="NAIC Consumer Shopping Tool for Auto Insurance",
        url=(
            "https://content.naic.org/sites/default/files/inline-files/"
            "topic_transparency_readability_consumer_auto_tool.pdf"
        ),
        state="NAIC",
        doc_type="guide",
        filename="naic-consumer-shopping-tool-auto.pdf",
    ),
    DocumentMeta(
        name="NAIC A Shopping Tool for Auto Insurance",
        url="https://content.naic.org/sites/default/files/consumer-auto-shopping-tool.pdf",
        state="NAIC",
        doc_type="guide",
        filename="naic-shopping-tool-auto-insurance.pdf",
    ),
]

BY_FILENAME: dict[str, DocumentMeta] = {doc.filename: doc for doc in DOCUMENTS}
