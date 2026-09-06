from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import finalize_core_gold_metadata as finalizer  # noqa: E402


class MetadataFinalizerTests(unittest.TestCase):
    def test_answer_types_are_clean(self) -> None:
        self.assertEqual(finalizer.answer_type("42"), "数值")
        self.assertEqual(finalizer.answer_type("March 10, 1949"), "日期或年份")
        self.assertEqual(finalizer.answer_type("Benedict Arnold"), "短实体")


if __name__ == "__main__":
    unittest.main()
