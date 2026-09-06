import unittest

import select_conflict_high_quality_v1 as target


class ConflictSelectorTests(unittest.TestCase):
    def test_normalized_direct_answer_match(self):
        self.assertTrue(target.normalized_equal("33%", "33"))

    def test_compound_roles_are_detected(self):
        self.assertIsNotNone(
            target.COMPOUND_ROLE.search(
                "The film was written, produced and directed by John Hughes."
            )
        )

    def test_document_id_is_not_semantic_target(self):
        self.assertIsNotNone(target.DOCUMENT_ID_TARGET.fullmatch("hp_doc_2"))

    def test_place_proximity_relation_is_detected(self):
        self.assertIsNotNone(
            target.PROXIMITY_RELATION.search(
                "South Somercotes is approximately two miles south from Saltfleetby."
            )
        )


if __name__ == "__main__":
    unittest.main()
