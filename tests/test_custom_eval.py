from __future__ import annotations

import unittest

from app.eval.evaluators import eval_source_rule
from app.eval.run_openevals_custom import eval_input_from_run


class CustomEvalContractTests(unittest.TestCase):
    def test_custom_eval_preserves_mismatched_source_ids(self):
        output = {
            "answer": "Grounded answer.",
            "sources": [{"id": "E1", "label": "Displayed source"}],
            "footnotes": {1: {"title": "Displayed source"}},
            "trace": {
                "local_context_preview": "Retrieved context.",
                "retrieval": {"evidence": [{"id": "E2"}]},
            },
        }
        inp = eval_input_from_run(
            {"question": "Question"},
            output,
            {"answer": "Reference"},
        )
        self.assertEqual(inp.sources, output["sources"])
        self.assertEqual(inp.retrieved_evidence_ids, ["E2"])
        self.assertEqual(eval_source_rule(inp)["score"], 0.0)

    def test_custom_eval_preserves_abstention(self):
        output = {
            "answer": "I keep personal details private.",
            "sources": [],
            "footnotes": {},
            "trace": {"refusal": True},
        }
        inp = eval_input_from_run(
            {"question": "What is your home address?"},
            output,
            {},
        )
        self.assertTrue(inp.abstained)


if __name__ == "__main__":
    unittest.main()
