from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

from app.agent.lg_controller import (
    LGController,
    _asks_private_question,
    _asks_for_unsupported_lime_detail,
    _web_evidence,
    hide_inline_citations,
    validate_answer,
)
from app.agent.rag_types import Evidence, RetrievalResult
from app.api import main as api_main
from app.api.main import _sse_event, healthz, home
from app.eval.evaluators import EvalInput, eval_faithfulness, eval_source_rule
from app.eval.run_eval import eval_input_from_output
from app.services.web_search import search_web


class _FakeController:
    def respond(self, question, on_token=None):
        if on_token:
            on_token("Grounded answer.")
        return {
            "answer": "Grounded answer.",
            "sources": [{"id": "E1", "label": "Approved source"}],
            "footnotes": {1: {"title": "Approved source"}},
            "source_freshness": {"index_built_at": "2026-07-27T00:00:00+00:00"},
            "validation": {"source_count": 1},
            "trace": {"latency_ms": 12.0},
        }


class ControllerApiTests(unittest.TestCase):
    def test_inline_citations_are_hidden_from_answer_text(self):
        answer = (
            "I built evaluation systems [1] and production RAG [E2]. "
            "Both were grounded in measured outcomes 【3†resume】."
        )
        self.assertEqual(
            hide_inline_citations(answer),
            "I built evaluation systems and production RAG. "
            "Both were grounded in measured outcomes.",
        )

    def test_citation_attribution_is_removed_without_malformed_prose(self):
        self.assertEqual(
            hide_inline_citations(
                "According to [1], the system reduced latency. "
                "The evidence, as shown in [E2], supports that result."
            ),
            "The system reduced latency. The evidence supports that result.",
        )
        self.assertEqual(
            hide_inline_citations("Per 【3†resume】, she led the evaluation."),
            "She led the evaluation.",
        )

    def test_production_controller_uses_lightweight_web_search_service(self):
        from app.agent import lg_controller

        self.assertIs(lg_controller.search_web, search_web)
        self.assertEqual(search_web.__module__, "app.services.web_search")

    def test_web_search_without_key_returns_structured_error(self):
        with patch.dict("os.environ", {}, clear=True):
            result = search_web("recent Lime announcements")

        self.assertEqual(result["results"], [])
        self.assertEqual(result["error"], "missing_tavily_api_key")

    def test_private_question_is_refused_without_sources(self):
        streamed = []
        result = LGController().respond(
            "What is your home address?",
            on_token=streamed.append,
        )
        self.assertIn("professional background", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["trace"]["refusal"])
        self.assertEqual("".join(streamed), result["answer"])

    def test_professional_model_family_question_is_not_private(self):
        question = "What family of models have you used for churn prediction?"
        self.assertFalse(_asks_private_question(question))
        self.assertFalse(
            _asks_private_question(
                "How has your family of models for churn prediction evolved?"
            )
        )
        controller = LGController()
        retrieval = RetrievalResult(
            evidence=[
                Evidence(
                    id="E1",
                    content="Used XGBoost for churn prediction.",
                    label="resume",
                    source_type="local_pdf",
                    score=0.03,
                )
            ],
            confidence="high",
        )
        with (
            patch.object(controller, "_retrieve", return_value=(retrieval, None)),
            patch.object(
                controller,
                "_compose",
                return_value=("I’ve used tree-based models such as XGBoost.", None),
            ) as compose,
        ):
            output = controller.respond(question)
        compose.assert_called_once()
        self.assertNotIn("refusal", output["trace"])

    def test_personal_web_results_allow_only_canonical_profile(self):
        payload = {
            "results": [
                {
                    "title": "Unapproved profile",
                    "url": "https://example.com/chivon",
                    "content": "Unsupported claims.",
                },
                {
                    "title": "LinkedIn",
                    "url": "https://www.linkedin.com/in/chivon-powers-phd-a6730610/",
                    "content": "Chivon Powers, PhD — Lime.",
                },
            ]
        }
        evidence = _web_evidence(
            "What is your current role?",
            payload,
            start_index=1,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].source_type, "public_profile")

    def test_public_lime_announcements_allow_general_web_evidence(self):
        evidence = _web_evidence(
            "Has Lime made any recent public announcements?",
            {
                "results": [
                    {
                        "title": "Lime announcement",
                        "url": "https://www.li.me/news/example",
                        "content": "A recent public announcement from Lime.",
                    }
                ]
            },
            start_index=1,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].source_type, "web")

    def test_validation_is_non_rewriting(self):
        result = validate_answer("A clear answer.", 2)
        self.assertFalse(result["rewrote_streamed_answer"])
        self.assertEqual(result["source_count"], 2)

    def test_high_confidence_local_route_skips_rewrite_and_web(self):
        local = RetrievalResult(
            evidence=[
                Evidence(
                    id="E1",
                    content="Approved local evidence.",
                    label="local • resume.pdf p.1",
                    source_type="local_pdf",
                    score=0.03,
                )
            ],
            confidence="high",
        )
        with (
            patch("app.agent.lg_controller.retrieve_hybrid", return_value=local),
            patch("app.agent.lg_controller.rewrite_queries") as rewrite,
            patch("app.agent.lg_controller.search_web") as web,
        ):
            result, payload = LGController()._retrieve("Tell me about Rocket Money")
        self.assertEqual(result.confidence, "high")
        self.assertIsNone(payload)
        rewrite.assert_not_called()
        web.assert_not_called()

    def test_only_unsupported_lime_details_are_deterministically_blocked(self):
        self.assertTrue(
            _asks_for_unsupported_lime_detail("Who is your manager at Lime?")
        )
        self.assertTrue(
            _asks_for_unsupported_lime_detail("Why did you decide to join Lime?")
        )
        self.assertTrue(
            _asks_for_unsupported_lime_detail("Which LLM did you fine-tune at Lime?")
        )
        self.assertFalse(
            _asks_for_unsupported_lime_detail("What is your title at Lime?")
        )
        self.assertFalse(
            _asks_for_unsupported_lime_detail(
                "Which projects are you leading at Lime?"
            )
        )
        self.assertFalse(
            _asks_for_unsupported_lime_detail(
                "How does your current chapter at Lime fit your career story?"
            )
        )

        controller = LGController()
        profile_result = RetrievalResult(
            evidence=[
                Evidence(
                    id="E1",
                    content="Chivon Powers is currently at Lime.",
                    label="public profile • LinkedIn",
                    source_type="public_profile",
                    score=0.03,
                    url="https://www.linkedin.com/in/chivon-powers-phd-a6730610",
                )
            ],
            confidence="high",
        )
        with (
            patch.object(controller, "_retrieve", return_value=(profile_result, None)),
            patch.object(controller, "_compose") as compose,
        ):
            output = controller.respond("Who is your manager at Lime?")
        compose.assert_not_called()
        self.assertIn("won’t guess", output["answer"])
        self.assertEqual(output["sources"], [])

    def test_indexed_resume_can_support_lime_role_details(self):
        controller = LGController()
        resume_result = RetrievalResult(
            evidence=[
                Evidence(
                    id="E1",
                    content=(
                        "Lime, May 2026 - Present. Senior Data Scientist, "
                        "Payments & Fraud."
                    ),
                    label="local • current resume p.1",
                    source_type="local_pdf",
                    score=0.03,
                    path="current-resume.pdf",
                    metadata={"employer": "Lime"},
                )
            ],
            confidence="high",
        )
        with (
            patch.object(controller, "_retrieve", return_value=(resume_result, None)),
            patch.object(
                controller,
                "_compose",
                return_value=(
                    "I’m a Senior Data Scientist focused on Payments & Fraud at Lime.",
                    None,
                ),
            ) as compose,
        ):
            output = controller.respond("What is your exact title at Lime?")
        compose.assert_called_once()
        self.assertEqual(output["sources"][0]["source_type"], "local_pdf")

    def test_lime_snapshot_remains_available_when_index_is_degraded(self):
        empty = RetrievalResult(
            evidence=[],
            confidence="low",
            reasons=["index_unavailable"],
        )
        with (
            patch("app.agent.lg_controller.retrieve_hybrid", return_value=empty),
            patch("app.agent.lg_controller.search_web") as web,
        ):
            result, payload = LGController()._retrieve("Where do you work currently?")
        self.assertIsNone(payload)
        web.assert_not_called()
        self.assertEqual(result.confidence, "medium")
        self.assertEqual(result.evidence[0].source_type, "public_profile")

    def test_sse_final_payload_can_include_sources(self):
        encoded = _sse_event("final", {"answer": "Hi", "sources": [{"id": "E1"}]})
        self.assertTrue(encoded.startswith("event: final\n"))
        data_line = encoded.splitlines()[1].removeprefix("data: ")
        self.assertEqual(json.loads(data_line)["sources"][0]["id"], "E1")

    def test_chat_and_stream_endpoints_preserve_source_contract(self):
        client = TestClient(api_main.app)
        with (
            patch.object(api_main, "_controller", _FakeController()),
            patch.object(api_main, "_index_ready", True),
        ):
            response = client.post("/chat", json={"question": "Question"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["sources"][0]["id"], "E1")

            streamed = client.get("/chat/stream", params={"question": "Question"})
            self.assertEqual(streamed.status_code, 200)
            self.assertIn("event: token", streamed.text)
            self.assertIn("event: final", streamed.text)
            self.assertIn('"sources": [{"id": "E1"', streamed.text)

    def test_health_and_home_expose_new_contract(self):
        status = healthz()
        self.assertIn(status["status"], {"ok", "degraded", "not_ready"})
        self.assertIn("index", status)
        page = home()
        self.assertNotIn("Currently at Lime", page)
        self.assertIn('id="sources"', page)
        self.assertIn("answer.textContent = payload.answer;", page)
        self.assertNotIn("payload.answer || answer.textContent", page)

    def test_eval_dataset_has_at_least_sixty_cases(self):
        with open("app/eval/qas.yaml", encoding="utf-8") as handle:
            rows = yaml.safe_load(handle)
        self.assertGreaterEqual(len(rows), 60)

    def test_eval_source_alignment_allows_source_free_abstentions(self):
        abstention = EvalInput(
            question="What is your home address?",
            answer="I keep interview answers focused on professional background.",
            context="",
            footnotes={},
            sources=[],
            retrieved_evidence_ids=[],
            abstained=True,
        )
        self.assertEqual(eval_source_rule(abstention)["score"], 1.0)
        self.assertEqual(eval_faithfulness(abstention)["score"], 1.0)

    def test_eval_source_alignment_rejects_unretrieved_ids(self):
        inp = EvalInput(
            question="Question",
            answer="Answer",
            context="Evidence",
            footnotes={1: {"title": "Source"}},
            sources=[{"id": "E1", "label": "Source"}],
            retrieved_evidence_ids=["E2"],
        )
        self.assertEqual(eval_source_rule(inp)["score"], 0.0)

    def test_production_eval_preserves_source_alignment_contract(self):
        output = {
            "answer": "Grounded answer.",
            "sources": [{"id": "E1", "label": "Displayed source"}],
            "footnotes": {1: {"title": "Displayed source"}},
            "trace": {
                "local_context_preview": "Retrieved context.",
                "retrieval": {"evidence": [{"id": "E2"}]},
            },
        }
        inp = eval_input_from_output("Question", None, output)
        self.assertEqual(inp.sources, output["sources"])
        self.assertEqual(inp.retrieved_evidence_ids, ["E2"])
        self.assertEqual(eval_source_rule(inp)["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
