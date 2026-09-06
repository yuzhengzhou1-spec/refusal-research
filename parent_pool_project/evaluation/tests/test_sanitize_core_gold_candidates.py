from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import sanitize_core_gold_candidates as sanitizer  # noqa: E402


class SanitizeCoreGoldTests(unittest.TestCase):
    def test_alias_leakage_is_normalized(self) -> None:
        self.assertTrue(sanitizer.phrase_in_text("The New-York", "Was New York involved?"))

    def test_partial_word_is_not_leakage(self) -> None:
        self.assertFalse(sanitizer.phrase_in_text("York", "Yorkshire is a county"))


if __name__ == "__main__":
    unittest.main()
