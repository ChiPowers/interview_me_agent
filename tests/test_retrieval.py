from __future__ import annotations

import unittest

from langchain_core.documents import Document

from app.agent.retrieval import (
    bm25_rank,
    footnotes_from_evidence,
    retrieve_hybrid,
    sources_from_evidence,
)


class _DocStore:
    def __init__(self, docs):
        self._dict = {str(index): doc for index, doc in enumerate(docs)}


class _FakeStore:
    def __init__(self, docs):
        self.docs = docs
        self.docstore = _DocStore(docs)

    def similarity_search_with_score(self, query, k):
        matches = [
            doc
            for doc in self.docs
            if any(token.lower() in doc.page_content.lower() for token in query.split())
        ]
        ordered = matches + [doc for doc in self.docs if doc not in matches]
        return [(doc, float(index + 1)) for index, doc in enumerate(ordered[:k])]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(
                page_content="Rocket Money billing retry logic saved more than $600,000 annually.",
                metadata={
                    "source": "/docs/resume.pdf",
                    "source_type": "local_pdf",
                    "label": "local • resume.pdf p.1",
                    "page_number": 1,
                    "employer": "Rocket Money",
                    "section": "experience",
                },
            ),
            Document(
                page_content="Butter payment routing experiments had an estimated $750k impact.",
                metadata={
                    "source": "/docs/projects.pdf",
                    "source_type": "local_pdf",
                    "label": "local • projects.pdf p.1",
                    "page_number": 1,
                    "employer": "Butter",
                    "section": "projects",
                },
            ),
            Document(
                page_content="Chivon Powers is currently at Lime.",
                metadata={
                    "source": "/docs/linkedin.json",
                    "source_type": "public_profile",
                    "label": "public profile • LinkedIn • current employer",
                    "url": "https://www.linkedin.com/in/chivon-powers-phd-a6730610",
                    "verified_at": "2026-07-27T00:00:00+00:00",
                    "employer": "Lime",
                    "section": "current_role",
                },
            ),
        ]

    def test_bm25_prioritizes_exact_business_evidence(self):
        ranked = bm25_rank("Rocket Money billing savings", self.docs)
        self.assertIs(ranked[0][0], self.docs[0])

    def test_hybrid_retrieval_returns_typed_diverse_evidence(self):
        result = retrieve_hybrid(
            "How did Rocket Money billing save money?",
            store=_FakeStore(self.docs),
            candidate_k=3,
            context_k=2,
        )
        self.assertEqual(result.evidence[0].metadata["employer"], "Rocket Money")
        self.assertLessEqual(len(result.evidence), 2)
        self.assertEqual(result.evidence[0].id, "E1")
        self.assertIn(result.confidence, {"high", "medium"})

    def test_named_employer_boost_keeps_lime_resume_in_context(self):
        lime_resume = Document(
            page_content=(
                "Senior Data Scientist, Payments & Fraud. Built a graph-based "
                "fraud patrol agent and identified $3M+ in revenue exposure."
            ),
            metadata={
                "source": "/docs/current-resume.pdf",
                "source_type": "local_pdf",
                "label": "local • current-resume.pdf p.1",
                "page_number": 1,
                "employer": "Lime",
                "section": "experience",
            },
        )
        docs = [*self.docs, lime_resume]
        result = retrieve_hybrid(
            "What measurable impact have you delivered at Lime?",
            store=_FakeStore(docs),
            candidate_k=4,
            context_k=2,
        )
        self.assertTrue(
            all(item.metadata["employer"] == "Lime" for item in result.evidence)
        )
        self.assertTrue(
            any(item.source_type == "local_pdf" for item in result.evidence)
        )
        self.assertIn("employer_metadata_boost=Lime", result.reasons)

    def test_explicit_employer_match_avoids_low_confidence_rewrite_route(self):
        result = retrieve_hybrid(
            "What measurable impact have you delivered at Lime?",
            store=_FakeStore(self.docs),
            candidate_k=3,
            context_k=1,
        )
        self.assertNotEqual(result.confidence, "low")
        self.assertEqual(result.evidence[0].metadata["employer"], "Lime")

    def test_sources_and_legacy_footnotes_share_same_evidence(self):
        result = retrieve_hybrid(
            "Where do you currently work?",
            store=_FakeStore(self.docs),
            candidate_k=3,
            context_k=2,
        )
        sources = sources_from_evidence(result.evidence)
        footnotes = footnotes_from_evidence(result.evidence)
        self.assertEqual(len(sources), len(footnotes))
        self.assertEqual(sources[0]["label"], footnotes[1]["title"])


if __name__ == "__main__":
    unittest.main()
