"""Local Ollama adapter selecting exact source spans by ID, never freehand quotes.

The original wire request, schema and response are retained alongside the
resolved validation representation. No network calls occur at import time.
"""
from __future__ import annotations

import copy
import json

from slr_pipeline import screening
from slr_pipeline.common import digest

VERSION = "source-ids-1"
BASE_CLIENT = screening.OllamaClient
INSTRUCTIONS = """
SOURCE CITATION INTERFACE: The schema uses citation_id instead of a quotation.
source_citations contains ALL original source text, split into labelled exact
spans. Select the ID whose text actually supports the specific judgement. The
software will insert that exact span and its original passage key. Do not
invent, paraphrase or type quotation text. For unclear evidence the ID may be
empty. For full_text investigated_data_property, operational_assessment and
full_text_sufficient, select a span whose passage_id begins BODY_. Check that
the selected span describes this report's own study rather than a cited study.
No earlier review decision has been supplied. This is still an AI proposal.
"""


def source_inventory(passages):
    inventory = {}
    for passage_id, text in passages.items():
        offset = 0
        while offset < len(text):
            end = min(len(text), offset + 700)
            if end < len(text):
                # Prefer a sentence/line boundary, keeping every character.
                stops = [text.rfind(token, offset + 200, end) for token in (". ", ".\n", "\n")]
                boundary = max(stops)
                if boundary >= offset + 200:
                    end = boundary + 1
                else:
                    boundary = text.rfind(" ", offset + 200, end)
                    if boundary >= offset + 200:
                        end = boundary + 1
            key = f"C{len(inventory) + 1:05d}"
            inventory[key] = {"passage_id": passage_id, "start": offset, "end": end, "text": text[offset:end]}
            offset = end
    return inventory


def citation_schema(schema, ids):
    value = copy.deepcopy(schema)
    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties", {})
            if "quote" in props and "passage_id" in props:
                del props["quote"], props["passage_id"]
                props["citation_id"] = {"type": "string", "enum": [""] + list(ids)}
                node["required"] = [x for x in node.get("required", []) if x not in {"quote", "passage_id"}] + ["citation_id"]
            for item in node.values(): walk(item)
        elif isinstance(node, list):
            for item in node: walk(item)
    walk(value)
    return value


def wire_request(messages, schema):
    if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise screening.AssessmentError("Source citation adapter expects one source request")
    evidence = json.loads(messages[1]["content"])
    inventory = source_inventory(evidence["passages"])
    payload = {k: v for k, v in evidence.items() if k != "passages"}
    payload["source_citations"] = inventory
    sent = [{"role": "system", "content": messages[0]["content"] + "\n" + INSTRUCTIONS},
            {"role": "user", "content": screening._json(payload)}]
    return sent, citation_schema(schema, inventory), inventory


def resolve_content(data, inventory):
    value = copy.deepcopy(data)
    for group in ("criteria", "extraction", "appraisal", "iso_mapping"):
        for item in value.get(group, []):
            key = item.pop("citation_id")
            if not isinstance(key, str) or (key and key not in inventory):
                raise screening.AssessmentError("Model selected an unknown source citation")
            source = inventory.get(key, {})
            item.update(passage_id=source.get("passage_id", ""), quote=source.get("text", ""))
    return value


class SourceCitationClient(BASE_CLIENT):
    def call(self, messages, schema):
        sent, selected_schema, inventory = wire_request(messages, schema)
        bound = sum(len(m["content"].encode("utf-8")) for m in sent) + 2048 + self.output
        if bound > self.context:
            raise screening.AssessmentError("Complete source citation inventory exceeds reserved local context")
        raw = super().call(sent, selected_schema)
        result = copy.deepcopy(raw)
        # The canonical content is a deterministic source lookup, not a model
        # generation. Preserve the ACTUAL generation/request as its provenance.
        result["source_citation_generation"] = {
            "adapter_version": VERSION, "wire_prompt_sha256": digest({"messages": sent, "schema": selected_schema}),
            "wire_messages": sent, "wire_schema": selected_schema, "raw_response": raw,
            "resolution": "citation IDs mapped to exact original source spans; semantic judgements unchanged",
        }
        try:
            resolved = resolve_content(json.loads(raw["message"]["content"]), inventory)
            result["message"]["content"] = screening._json(resolved)
        except (KeyError, TypeError, ValueError, screening.AssessmentError) as exc:
            # Preserve an invalid original response as well. The unchanged
            # validator will reject this canonical error object, rather than
            # accepting or discarding the failed generation.
            result["source_citation_generation"]["resolution_error"] = type(exc).__name__
            result["message"] = {"role": "assistant", "content": screening._json({"citation_resolution_error": type(exc).__name__})}
        return result
