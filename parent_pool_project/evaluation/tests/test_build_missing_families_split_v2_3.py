import unittest

import build_missing_families_split_v2_3 as target


def sample():
    return {
        "样本ID": "demo-1",
        "来源": "demo",
        "问题": "Which NBA team did Donald Sterling own from 1981 through 2014?",
        "答案": ["Los Angeles Clippers", "Clippers"],
        "完整上下文": [
            {
                "文档ID": "doc1",
                "标题": "Ownership",
                "文本": "Donald Sterling owned the team from 1981 through 2014. He sold it later.",
            },
            {
                "文档ID": "doc2",
                "标题": "Person",
                "文本": "Donald Sterling was a former owner of the team.",
            },
        ],
        "核心黄金筛选": {
            "query_only_knowledge_clean": True,
            "clean_full_pass": True,
            "required_models": ["qwen3_8b"],
            "stages": {
                "QUERY_ONLY": {"qwen3_8b": {}},
                "CLEAN_FULL": {"qwen3_8b": {}},
            },
        },
    }


def valid_edit():
    return {
        "constructible": True,
        "skip_reason": "",
        "clarification_target": "the NBA team owned by Donald Sterling from 1981 through 2014",
        "restoring_information": "Donald Sterling owned the team from 1981 through 2014.",
        "edit_type": "REMOVE_RELATION",
        "edited_document_id": "doc1",
        "original_span": "Donald Sterling owned the team from 1981 through 2014.",
        "replacement_span": "",
        "modified_evidence": [
            {
                "document_id": "doc1",
                "title": "Ownership",
                "text": "He sold it later.",
            },
            {
                "document_id": "doc2",
                "title": "Person",
                "text": "Donald Sterling was a former owner of the team.",
            },
        ],
    }


def valid_clarification(text):
    return {
        "target_resolved_by_remaining_evidence": False,
        "resolution_reason": "",
        "known_support_document_id": "doc2",
        "known_support_span": "Donald Sterling was a former owner of the team.",
        "assistant_clarification": text,
    }


class MissingFamilySplitTests(unittest.TestCase):
    def test_retry_filter_drops_only_api_errors(self):
        calls = {
            "a": {"status": "SUCCESS"},
            "b": {"status": "API_ERROR"},
            "c": {"status": "STRUCTURE_ERROR"},
        }
        self.assertEqual(target.drop_retryable_api_errors(calls), ["b"])
        self.assertEqual(set(calls), {"a", "c"})

    def test_valid_minimal_edit(self):
        self.assertEqual(target.edit_errors(sample(), valid_edit()), [])

    def test_answer_entity_may_remain_in_unchanged_evidence(self):
        edit = valid_edit()
        self.assertIn("Donald Sterling", edit["modified_evidence"][1]["text"])
        self.assertEqual(target.edit_errors(sample(), edit), [])

    def test_surface_artifact_is_rejected(self):
        edit = valid_edit()
        edit["modified_evidence"][0]["text"] = "Donald Sterling worked with . He sold it later."
        errors = target.edit_errors(sample(), edit)
        self.assertTrue(any("surface artifact" in item for item in errors))

    def test_repeated_sentence_is_rejected(self):
        edit = valid_edit()
        edit["modified_evidence"][0]["text"] = (
            "He sold it later. He sold it later."
        )
        errors = target.edit_errors(sample(), edit)
        self.assertTrue(any("repeated sentence" in item for item in errors))

    def test_prompt_encodes_claim_not_keyword_policy(self):
        prompt = target.EDITOR_SYSTEM_PROMPT
        self.assertIn("may still mention the answer entity", prompt)
        self.assertIn("Do not clean all answer occurrences", prompt)
        self.assertIn("Do not merely erase an answer string", prompt)
        self.assertIn("not a stronger inference", prompt)
        self.assertIn("deleting only an award/result predicate is ineffective", prompt)

    def test_clarification_answer_leak_is_rejected(self):
        result = {"assistant_clarification": "Was it the Los Angeles Clippers?"}
        errors = target.clarification_errors(sample(), result, valid_edit())
        self.assertIn("assistant_clarification leaks an accepted answer", errors)

    def test_answer_is_forbidden_in_clarification_target(self):
        edit = valid_edit()
        edit["clarification_target"] = "whether Los Angeles Clippers was the team"
        errors = target.edit_errors(sample(), edit)
        self.assertIn("clarification_target contains an accepted answer", errors)

    def test_restoration_cannot_add_stronger_relation(self):
        edit = valid_edit()
        edit["clarification_target"] = "the county seat associated with the team"
        edit["restoring_information"] = "The team is the county seat."
        errors = target.edit_errors(sample(), edit)
        self.assertIn(
            "sensitive target relation is absent from original_span", errors
        )

    def test_clarification_must_cover_fixed_target(self):
        result = {"assistant_clarification": "Which county contains Indian Creek?"}
        errors = target.clarification_errors(sample(), result, valid_edit())
        self.assertIn("assistant_clarification changes the target relation", errors)

    def test_evidence_aware_three_part_clarification_is_valid(self):
        result = valid_clarification(
                "The context identifies Donald Sterling as a former NBA franchise "
                "owner, but it does not specify which team he owned from 1981 "
                "through 2014. Could you provide the team name?"
        )
        self.assertEqual(
            target.clarification_errors(sample(), result, valid_edit()), []
        )

    def test_clarifier_view_hides_answer_bearing_document(self):
        row = sample()
        edit = valid_edit()
        edit["modified_evidence"][1]["text"] = (
            "Donald Sterling was a former Los Angeles Clippers owner. "
            "He was involved in NBA litigation."
        )
        safe = target.clarification_safe_evidence(row, edit)
        self.assertNotIn("Los Angeles Clippers", safe[1]["text"])
        self.assertEqual(safe[1]["text"], "")

    def test_clarifier_view_removes_name_with_middle_token(self):
        row = sample()
        row["答案"] = ["Larry Drake"]
        edit = valid_edit()
        edit["modified_evidence"][1]["text"] = (
            "Larry Richard Drake was an American actor. Another clue remains."
        )
        safe = target.clarification_safe_evidence(row, edit)
        self.assertNotIn("Larry Richard Drake", safe[1]["text"])
        self.assertEqual(safe[1]["text"], "")

    def test_known_support_span_must_exist_exactly(self):
        result = valid_clarification(
            "The context identifies Donald Sterling as a former NBA owner, but "
            "it does not identify the team. Could you provide the team name?"
        )
        result["known_support_span"] = "Donald Sterling owned an NBA franchise."
        errors = target.clarification_errors(sample(), result, valid_edit())
        self.assertIn(
            "known_support_span is not an exact remaining-evidence span", errors
        )

    def test_already_resolved_target_is_a_valid_rejection_payload(self):
        result = {
            "target_resolved_by_remaining_evidence": True,
            "resolution_reason": "The remaining text directly names the team.",
            "known_support_document_id": "",
            "known_support_span": "",
            "assistant_clarification": "",
        }
        self.assertEqual(
            target.clarification_errors(sample(), result, valid_edit()), []
        )

    def test_clarifier_messages_do_not_contain_accepted_answer(self):
        row = sample()
        edit = valid_edit()
        edit["modified_evidence"][1]["text"] = (
            "Donald Sterling was a former Los Angeles Clippers owner."
        )
        messages = target.clarification_messages(row, edit)
        self.assertNotIn("Los Angeles Clippers", messages[1]["content"])

    def test_mechanical_question_repetition_is_rejected(self):
        result = {
            "assistant_clarification": (
                "Which NBA team was owned by Donald Sterling from 1981 to 2014?"
            )
        }
        errors = target.clarification_errors(sample(), result, valid_edit())
        self.assertIn(
            "assistant_clarification lacks known-support framing", errors
        )
        self.assertIn(
            "assistant_clarification lacks support-gap contrast", errors
        )

    def test_incomplete_list_must_not_be_treated_as_exhaustive(self):
        result = valid_clarification(
            "The context lists several actors in the film, but it does not identify "
            "which of those actors appeared in L.A. Law. Could you name the actor?"
        )
        errors = target.clarification_errors(sample(), result, valid_edit())
        self.assertIn(
            "assistant_clarification assumes an incomplete list contains the answer",
            errors,
        )

    def test_distance_from_is_not_country_origin(self):
        edit = valid_edit()
        edit["clarification_target"] = "the distance of the team from Sydney"
        edit["restoring_information"] = "The team is 200 kilometres from Sydney."
        edit["original_span"] = "The team is 200 kilometres from Sydney."
        edit["modified_evidence"][0]["text"] = "He sold it later."
        errors = target.edit_errors(sample(), edit)
        self.assertFalse(any("origin relation" in item for item in errors))

    def test_event_answer_cannot_remain_in_replacement(self):
        row = sample()
        row["答案"] = ["1987 Pan American Games"]
        edit = valid_edit()
        edit["clarification_target"] = "the event where Matt Scoggin won silver"
        edit["original_span"] = (
            "Matt Scoggin won silver at the 1987 Pan American Games."
        )
        edit["replacement_span"] = (
            "Matt Scoggin competed at the 1987 Pan American Games."
        )
        edit["restoring_information"] = edit["original_span"]
        edit["modified_evidence"][0]["text"] = edit["replacement_span"]
        errors = target.edit_errors(row, edit)
        self.assertIn(
            "accepted event remains explicitly identified in replacement_span", errors
        )

    def test_relation_preserving_but_context_free_clarification_is_rejected(self):
        edit = valid_edit()
        edit["clarification_target"] = "the founder of the publisher of a journal"
        result = {"assistant_clarification": "Who founded Routledge?"}
        errors = target.clarification_errors(sample(), result, edit)
        self.assertIn(
            "assistant_clarification lacks known-support framing", errors
        )

    def test_clarification_cannot_mention_evidence(self):
        result = {
            "assistant_clarification": (
                "The evidence is missing the NBA team owned by Donald Sterling; "
                "which team was it?"
            )
        }
        errors = target.clarification_errors(sample(), result, valid_edit())
        self.assertIn("assistant_clarification mentions evidence manipulation", errors)

    def test_film_editing_role_is_not_evidence_manipulation(self):
        edit = valid_edit()
        edit["clarification_target"] = "the director of the film"
        result = {
            "assistant_clarification": (
                "The context establishes that the film was co-written and edited, "
                "but it does not identify its director. Could you provide the "
                "director's name?"
            )
        }
        errors = target.clarification_errors(sample(), result, edit)
        self.assertNotIn(
            "assistant_clarification mentions evidence manipulation", errors
        )

    def test_clarification_cannot_ask_two_slots(self):
        edit = valid_edit()
        edit["clarification_target"] = "the founder of the publisher of a journal"
        result = {
            "assistant_clarification": (
                "Which publisher is meant, and who founded that publisher?"
            )
        }
        errors = target.clarification_errors(sample(), result, edit)
        self.assertIn(
            "assistant_clarification asks multiple information slots", errors
        )

    def test_restoration_and_final_answer_are_deterministic(self):
        edit_call = {
            "parsed_response": valid_edit(),
            "model_alias": "deepseek_v4_pro",
            "requested_model_id": "deepseek-v4-pro",
            "request_body": {"temperature": 0},
            "prompt_version": "edit",
            "request_hash": "e",
            "attempts": [],
        }
        clarify_call = {
            "parsed_response": valid_clarification("Who owned it during that period?"),
            "prompt_version": "clarify",
            "request_hash": "c",
            "attempts": [],
        }
        family = target.build_family(sample(), edit_call, clarify_call)
        self.assertEqual(
            family["trajectory"][1]["content"],
            family["restoration"]["restoring_information"],
        )
        self.assertEqual(family["trajectory"][2]["content"], "Los Angeles Clippers")


if __name__ == "__main__":
    unittest.main()
