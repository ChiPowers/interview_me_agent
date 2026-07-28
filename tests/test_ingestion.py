from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from app.services import ingest_index
from app.services.profile_snapshot import refresh_profile_snapshot


class IngestionTests(unittest.TestCase):
    def test_sanitize_professional_text_removes_contact_details(self):
        text = (
            "Contact person@example.com or (267) 975-6794. "
            "Mail: 123 Main Street, Denver, CO 80122."
        )
        clean = ingest_index.sanitize_professional_text(text)
        self.assertNotIn("person@example.com", clean)
        self.assertNotIn("267", clean)
        self.assertNotIn("123 Main Street", clean)
        self.assertNotIn("80122", clean)

    def test_structure_metadata_labels_employer_and_topic(self):
        doc = Document(
            page_content="Rocket Money churn model and A/B experimentation work.",
            metadata={"source": "/tmp/resume.pdf", "page": 0, "source_type": "local_pdf"},
        )
        chunks = ingest_index._split_and_label([doc])
        self.assertEqual(chunks[0].metadata["employer"], "Rocket Money")
        self.assertIn("experimentation", chunks[0].metadata["topics"])
        self.assertEqual(chunks[0].metadata["page_number"], 1)

    def test_manifest_invalidates_when_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "raw"
            pdf_dir.mkdir()
            pdf = pdf_dir / "resume.pdf"
            pdf.write_bytes(b"version one")
            profile = root / "linkedin.json"
            profile.write_text("{}", encoding="utf-8")
            manifest_path = root / "manifest.json"

            with patch.object(ingest_index, "PROFILE_SNAPSHOT_PATH", profile):
                ingest_index.write_manifest(pdf_dir, 1, manifest_path)
                self.assertTrue(
                    ingest_index.manifest_is_current(pdf_dir, manifest_path)
                )
                pdf.write_bytes(b"version two")
                self.assertFalse(
                    ingest_index.manifest_is_current(pdf_dir, manifest_path)
                )

    def test_manifest_is_portable_across_checkout_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_raw = root / "checkout-a" / "data" / "raw"
            second_raw = root / "checkout-b" / "data" / "raw"
            first_raw.mkdir(parents=True)
            second_raw.mkdir(parents=True)
            (first_raw / "resume.pdf").write_bytes(b"same resume")
            (second_raw / "resume.pdf").write_bytes(b"same resume")
            first_profile = root / "checkout-a" / "app" / "data" / "linkedin.json"
            second_profile = root / "checkout-b" / "app" / "data" / "linkedin.json"
            first_profile.parent.mkdir(parents=True)
            second_profile.parent.mkdir(parents=True)
            first_profile.write_text('{"employer":"Lime"}', encoding="utf-8")
            second_profile.write_text('{"employer":"Lime"}', encoding="utf-8")
            manifest_path = root / "manifest.json"

            with patch.object(
                ingest_index, "PROFILE_SNAPSHOT_PATH", first_profile
            ):
                manifest = ingest_index.write_manifest(
                    first_raw, 1, manifest_path
                )
            self.assertEqual(
                set(manifest["sources"]),
                {"pdf/resume.pdf", "profile/linkedin.json"},
            )

            with patch.object(
                ingest_index, "PROFILE_SNAPSHOT_PATH", second_profile
            ):
                self.assertTrue(
                    ingest_index.manifest_is_current(second_raw, manifest_path)
                )

    def test_failed_profile_refresh_retains_last_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            previous = {
                "canonical_url": "https://example.com/profile",
                "facts": [{"field": "current_employer", "value": "Lime"}],
            }
            path.write_text(json.dumps(previous), encoding="utf-8")
            result = refresh_profile_snapshot(
                path=path,
                url="https://example.com/profile",
                fetcher=lambda _url: "",
            )
            self.assertEqual(result, previous)
            self.assertEqual(json.loads(path.read_text()), previous)


if __name__ == "__main__":
    unittest.main()
