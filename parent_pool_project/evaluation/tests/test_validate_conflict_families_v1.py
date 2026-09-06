import unittest

import validate_conflict_families_v1 as target


class ConflictSemanticGateTests(unittest.TestCase):
    def test_consistent_pass_is_valid(self):
        result = {
            "valid": True,
            "same_atomic_fact": True,
            "single_disputed_slot": True,
            "mutually_exclusive": True,
            "necessary_for_query": True,
            "changes_query_answer": True,
            "not_resolvable_without_user": True,
            "resolution_restores_answer": True,
            "failure_codes": [],
            "reason": "The claims assign different directors to the same film.",
        }
        self.assertEqual(target.audit_errors(result), [])

    def test_derived_verdict_cannot_hide_failed_dimension(self):
        result = {
            "valid": True,
            "same_atomic_fact": True,
            "single_disputed_slot": True,
            "mutually_exclusive": False,
            "necessary_for_query": True,
            "changes_query_answer": True,
            "not_resolvable_without_user": True,
            "resolution_restores_answer": True,
            "failure_codes": [],
            "reason": "Guest lists can contain both names.",
        }
        self.assertEqual(target.audit_errors(result), [])
        self.assertFalse(target.derived_valid(result))

    def test_programmatic_verdict_ignores_model_verdict_and_codes(self):
        result = {
            "valid": True,
            "same_atomic_fact": True,
            "single_disputed_slot": True,
            "mutually_exclusive": True,
            "necessary_for_query": True,
            "changes_query_answer": True,
            "not_resolvable_without_user": True,
            "resolution_restores_answer": True,
            "failure_codes": ["CORRECTION_FRAMING"],
            "reason": "The appended sentence is framed as a correction.",
        }
        self.assertEqual(target.audit_errors(result), [])
        self.assertTrue(target.derived_valid(result))

    def test_independently_resolvable_conflict_fails(self):
        result = {
            "valid": False,
            "same_atomic_fact": True,
            "single_disputed_slot": True,
            "mutually_exclusive": True,
            "necessary_for_query": True,
            "changes_query_answer": True,
            "not_resolvable_without_user": False,
            "resolution_restores_answer": True,
            "failure_codes": ["not_resolvable_without_user"],
            "reason": "The component counts independently determine the total.",
        }
        self.assertEqual(target.audit_errors(result), [])
        self.assertFalse(target.derived_valid(result))

    def test_multiple_changed_roles_fail(self):
        result = {
            "valid": False,
            "same_atomic_fact": True,
            "single_disputed_slot": False,
            "mutually_exclusive": True,
            "necessary_for_query": True,
            "changes_query_answer": True,
            "not_resolvable_without_user": True,
            "resolution_restores_answer": True,
            "failure_codes": ["single_disputed_slot"],
            "reason": "The substituted person fills writer, producer, and director roles.",
        }
        self.assertEqual(target.audit_errors(result), [])
        self.assertFalse(target.derived_valid(result))


if __name__ == "__main__":
    unittest.main()
