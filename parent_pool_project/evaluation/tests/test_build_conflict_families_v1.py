import unittest

import build_conflict_families_v1 as target


def sample():
    return {
        "样本ID": "demo-conflict-1",
        "来源": "demo",
        "问题": "What year was Pull Up to the Bumper released?",
        "答案": ["1981"],
        "完整上下文": [
            {
                "文档ID": "doc1",
                "标题": "Pull Up to the Bumper",
                "文本": (
                    '"Pull Up to the Bumper" is a 1981 single by Jamaican '
                    'singer Grace Jones.'
                ),
            },
            {
                "文档ID": "doc2",
                "标题": "Grace Jones",
                "文本": "Grace Jones is a Jamaican singer and actress.",
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
    original = (
        '"Pull Up to the Bumper" is a 1981 single by Jamaican singer Grace Jones.'
    )
    conflict = (
        '"Pull Up to the Bumper" is a 1983 single by Jamaican singer Grace Jones.'
    )
    return {
        "constructible": True,
        "skip_reason": "",
        "conflict_target": "the release year of Pull Up to the Bumper",
        "edited_document_id": "doc1",
        "original_claim": original,
        "original_value": "1981",
        "conflicting_claim": conflict,
        "conflicting_value": "1983",
        "necessity_explanation": "The release year directly answers the query.",
        "exclusivity_explanation": "One release cannot have both years.",
        "modified_evidence": [
            {
                "document_id": "doc1",
                "title": "Pull Up to the Bumper",
                "text": f"{original} {conflict}",
            },
            {
                "document_id": "doc2",
                "title": "Grace Jones",
                "text": "Grace Jones is a Jamaican singer and actress.",
            },
        ],
    }


class ConflictFamilyTests(unittest.TestCase):
    def test_valid_atomic_append(self):
        self.assertEqual(target.edit_errors(sample(), valid_edit()), [])

    def test_original_document_cannot_be_rewritten(self):
        edit = valid_edit()
        edit["modified_evidence"][0]["text"] = (
            "The song was released in 1981. " + edit["conflicting_claim"]
        )
        errors = target.edit_errors(sample(), edit)
        self.assertIn(
            "original document text was changed before the appended claim", errors
        )

    def test_conflicting_value_cannot_equal_answer(self):
        edit = valid_edit()
        edit["conflicting_value"] = "1981"
        edit["conflicting_claim"] = edit["original_claim"]
        edit["modified_evidence"][0]["text"] = (
            edit["original_claim"] + " " + edit["conflicting_claim"]
        )
        errors = target.edit_errors(sample(), edit)
        self.assertIn("conflicting_value matches an accepted answer", errors)

    def test_hedged_conflict_is_rejected(self):
        edit = valid_edit()
        edit["conflicting_claim"] = (
            "Pull Up to the Bumper was reportedly released in 1983."
        )
        edit["modified_evidence"][0]["text"] = (
            edit["original_claim"] + " " + edit["conflicting_claim"]
        )
        errors = target.edit_errors(sample(), edit)
        self.assertIn("conflicting_claim is hedged or framed as a correction", errors)

    def test_correction_framing_is_rejected(self):
        edit = valid_edit()
        edit["conflicting_claim"] = (
            "Pull Up to the Bumper was actually released in 1983."
        )
        edit["modified_evidence"][0]["text"] = (
            edit["original_claim"] + " " + edit["conflicting_claim"]
        )
        errors = target.edit_errors(sample(), edit)
        self.assertIn(
            "conflicting_claim is hedged or framed as a correction", errors
        )

    def test_incidental_edit_that_preserves_answer_is_rejected(self):
        value = sample()
        value["问题"] = "What series includes this film?"
        value["答案"] = ["DC Universe Animated Original Movies"]
        value["完整上下文"][0] = {
            "文档ID": "doc1",
            "标题": "Film",
            "文本": (
                "The film is the twenty-sixth film in the DC Universe Animated "
                "Original Movies series."
            ),
        }
        edit = valid_edit()
        edit.update(
            {
                "edited_document_id": "doc1",
                "original_claim": value["完整上下文"][0]["文本"],
                "original_value": "twenty-sixth",
                "conflicting_claim": (
                    "The film is the twenty-seventh film in the DC Universe "
                    "Animated Original Movies series."
                ),
                "conflicting_value": "twenty-seventh",
            }
        )
        edit["modified_evidence"] = [
            {
                "document_id": "doc1",
                "title": "Film",
                "text": edit["original_claim"] + " " + edit["conflicting_claim"],
            },
            {
                "document_id": "doc2",
                "title": "Grace Jones",
                "text": "Grace Jones is a Jamaican singer and actress.",
            },
        ]
        self.assertIn(
            "accepted answer remains unchanged in both claims",
            target.edit_errors(value, edit),
        )

    def test_clarification_must_present_both_values(self):
        result = {
            "assistant_conflict_resolution": (
                "The release year is listed as both 1981 and 1983, which is correct?"
            )
        }
        self.assertEqual(
            target.clarification_errors(sample(), result, valid_edit()), []
        )

    def test_clarification_omitting_value_is_rejected(self):
        result = {
            "assistant_conflict_resolution": (
                "The release year has two conflicting values; which is correct?"
            )
        }
        errors = target.clarification_errors(sample(), result, valid_edit())
        self.assertIn("assistant_conflict_resolution omits original_value", errors)
        self.assertIn("assistant_conflict_resolution omits conflicting_value", errors)

    def test_clarification_source_discrepancy_is_accepted(self):
        result = {
            "assistant_conflict_resolution": (
                "One source lists the release year as 1981, while another source "
                "lists it as 1983; which year is correct?"
            )
        }
        self.assertEqual(
            target.clarification_errors(sample(), result, valid_edit()), []
        )

    def test_shortened_subject_reference_is_accepted(self):
        value = sample()
        value["问题"] = "Who became president?"
        value["答案"] = ["Lenin Moreno"]
        value["完整上下文"][0] = {
            "文档ID": "doc1",
            "标题": "Lenin Moreno",
            "文本": "Moreno became president in 2017.",
        }
        edit = {
            "constructible": True,
            "skip_reason": "",
            "conflict_target": "the office assumed by Lenin Moreno in 2017",
            "edited_document_id": "doc1",
            "original_claim": "Moreno became president in 2017.",
            "original_value": "Lenin Moreno",
            "conflicting_claim": "Guillermo Lasso became president in 2017.",
            "conflicting_value": "Guillermo Lasso",
            "necessity_explanation": "The office holder answers the query.",
            "exclusivity_explanation": "Only one person won the fixed election.",
            "modified_evidence": [
                {
                    "document_id": "doc1",
                    "title": "Lenin Moreno",
                    "text": (
                        "Moreno became president in 2017. "
                        "Guillermo Lasso became president in 2017."
                    ),
                },
                {
                    "document_id": "doc2",
                    "title": "Grace Jones",
                    "text": "Grace Jones is a Jamaican singer and actress.",
                },
            ],
        }
        self.assertEqual(target.edit_errors(value, edit), [])

    def test_pronoun_reference_supported_by_document_title(self):
        value = sample()
        value["问题"] = "Who was governor-general?"
        value["答案"] = ["Michael Ogio"]
        value["完整上下文"][0] = {
            "文档ID": "doc1",
            "标题": "Michael Ogio",
            "文本": "He served as governor-general from 2010.",
        }
        edit = {
            "constructible": True,
            "skip_reason": "",
            "conflict_target": "the office held from 2010",
            "edited_document_id": "doc1",
            "original_claim": "He served as governor-general from 2010.",
            "original_value": "Michael Ogio",
            "conflicting_claim": "Jeffery Nape served as governor-general from 2010.",
            "conflicting_value": "Jeffery Nape",
            "necessity_explanation": "The office holder answers the query.",
            "exclusivity_explanation": "The ordinal office holder is unique.",
            "modified_evidence": [
                {
                    "document_id": "doc1",
                    "title": "Michael Ogio",
                    "text": (
                        "He served as governor-general from 2010. "
                        "Jeffery Nape served as governor-general from 2010."
                    ),
                },
                {
                    "document_id": "doc2",
                    "title": "Grace Jones",
                    "text": "Grace Jones is a Jamaican singer and actress.",
                },
            ],
        }
        self.assertEqual(target.edit_errors(value, edit), [])

    def test_family_resolution_and_answer_are_deterministic(self):
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
            "parsed_response": {
                "assistant_conflict_resolution": (
                    "The year is listed as 1981 and 1983; which is correct?"
                )
            },
            "prompt_version": "clarify",
            "request_hash": "c",
            "attempts": [],
        }
        family = target.build_family(sample(), edit_call, clarify_call)
        self.assertEqual(
            family["trajectory"][1]["content"],
            "For the release year of Pull Up to the Bumper, the correct value is 1981.",
        )
        self.assertEqual(family["trajectory"][2]["content"], "1981")

    def test_prompt_requires_original_claim_to_remain(self):
        self.assertIn("Keep the original correct claim present", target.EDITOR_SYSTEM_PROMPT)
        self.assertIn("append exactly one new sentence", target.EDITOR_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
