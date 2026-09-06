import unittest

import validate_missing_families_v1 as target


def result(**overrides):
    value = {
        "valid": True,
        "deletion_effective": True,
        "known_support_grounded": True,
        "query_not_used_as_evidence": True,
        "gap_matches_edit": True,
        "restoration_recovers": True,
        "failure_codes": [],
        "reason": "The remaining evidence lacks the requested ownership relation.",
    }
    value.update(overrides)
    return value


class MissingSemanticGateTests(unittest.TestCase):
    def test_consistent_pass(self):
        value = result()
        self.assertEqual(target.audit_errors(value), [])
        self.assertTrue(target.derived_valid(value))

    def test_surviving_answer_chain_fails(self):
        value = result(
            valid=False,
            deletion_effective=False,
            failure_codes=["DELETION_INEFFECTIVE"],
        )
        self.assertEqual(target.audit_errors(value), [])
        self.assertFalse(target.derived_valid(value))

    def test_query_only_known_support_fails(self):
        value = result(
            valid=False,
            known_support_grounded=False,
            query_not_used_as_evidence=False,
            failure_codes=["QUERY_USED_AS_EVIDENCE"],
        )
        self.assertFalse(target.derived_valid(value))

    def test_programmatic_verdict_uses_dimensions(self):
        value = result(valid=False, failure_codes=["MODEL_LABEL_NOISE"])
        self.assertEqual(target.audit_errors(value), [])
        self.assertTrue(target.derived_valid(value))


if __name__ == "__main__":
    unittest.main()
