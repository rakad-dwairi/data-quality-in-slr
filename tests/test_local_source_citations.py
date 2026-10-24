"""Lossless source-ID citations with fake responses; never real inference."""
import json
import unittest
from unittest.mock import patch

import local_source_citations as citations
from slr_pipeline import screening


class SourceCitationTests(unittest.TestCase):
    def test_inventory_preserves_every_character_and_exact_source_offsets(self):
        passages = {"TITLE": "Synthetic title", "BODY_00000000_00006000": ("Alpha synthetic sentence.\n" * 100) + "Tail."}
        inventory = citations.source_inventory(passages)
        for pid, text in passages.items():
            parts = [x for x in inventory.values() if x["passage_id"] == pid]
            self.assertEqual("".join(p["text"] for p in parts), text)
            for part in parts:
                self.assertEqual(part["text"], text[part["start"]:part["end"]])
                self.assertLessEqual(len(part["text"]), 700)

    def test_schema_uses_only_enumerated_source_ids_and_keeps_original_unchanged(self):
        schema = screening._schema("full_text", list(screening.CORE_CRITERIA))
        original = json.dumps(schema, sort_keys=True)
        converted = citations.citation_schema(schema, ["C00001", "C00002"])
        item = converted["properties"]["criteria"]["items"]
        self.assertNotIn("quote", item["properties"])
        self.assertNotIn("passage_id", item["required"])
        self.assertEqual(item["properties"]["citation_id"]["enum"], ["", "C00001", "C00002"])
        self.assertEqual(json.dumps(schema, sort_keys=True), original)

    def test_resolution_changes_only_citations_not_decisions_attribution_or_fields(self):
        inventory = {"C00001": {"passage_id": "BODY_SYNTHETIC", "text": "Original source phrase."}}
        data = {"decision": "uncertain", "criteria": [{"criterion": "synthetic", "value": "yes", "citation_id": "C00001", "source_attribution": "prior_work"}],
                "extraction": [{"citation_id": "C00001", "fields": {"citation_id": "A literal domain field"}}]}
        resolved = citations.resolve_content(data, inventory)
        self.assertEqual(resolved["decision"], "uncertain")
        self.assertEqual(resolved["criteria"][0]["source_attribution"], "prior_work")
        self.assertEqual(resolved["criteria"][0]["quote"], "Original source phrase.")
        self.assertEqual(resolved["extraction"][0]["fields"]["citation_id"], "A literal domain field")
        self.assertIn("citation_id", data["criteria"][0])
        with self.assertRaises(screening.AssessmentError):
            citations.resolve_content({"criteria": [{"citation_id": "UNKNOWN"}]}, inventory)

    def test_wire_request_sends_all_source_text_without_prior_decisions(self):
        evidence = {"record_id": "R_SYNTHETIC", "passages": {"TITLE": "Synthetic title", "BODY_A": "Synthetic method."}}
        messages = [{"role": "system", "content": "Trusted synthetic rules"}, {"role": "user", "content": json.dumps(evidence)}]
        sent, schema, inventory = citations.wire_request(messages, screening._schema("title_abstract", ["synthetic"]))
        wire = json.loads(sent[1]["content"])
        self.assertNotIn("passages", wire)
        self.assertEqual(wire["source_citations"], inventory)
        self.assertNotIn("ai_decision", wire)
        self.assertEqual(evidence["passages"]["BODY_A"], "Synthetic method.")

    def test_client_preserves_actual_request_and_raw_id_response_on_success_and_failure(self):
        messages = [{"role": "system", "content": "Synthetic rules"}, {"role": "user", "content": json.dumps({"passages": {"BODY_A": "Synthetic method."}})}]
        schema = screening._schema("title_abstract", ["synthetic"])
        for key, valid in [("C00001", True), ("UNKNOWN", False)]:
            raw = {"done": True, "message": {"content": json.dumps({"criteria": [{"citation_id": key}], "decision": "uncertain"})}}
            with patch.object(citations.BASE_CLIENT, "call", return_value=raw) as call:
                client = citations.SourceCitationClient({"name": "synthetic", "context_tokens": 32768, "max_output_tokens": 4096})
                try:
                    result = client.call(messages, schema)
                finally:
                    client.close()
            self.assertEqual(result["source_citation_generation"]["raw_response"], raw)
            self.assertEqual(result["source_citation_generation"]["wire_messages"], call.call_args.args[0])
            resolved = json.loads(result["message"]["content"])
            if valid:
                self.assertEqual(resolved["criteria"][0]["quote"], "Synthetic method.")
            else:
                self.assertIn("citation_resolution_error", resolved)


if __name__ == "__main__":
    unittest.main()
