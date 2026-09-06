from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import answer_is_correct, parse_response, read_jsonl  # noqa: E402


class ProcessedDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = ROOT / "data" / "processed" / "family_100_v1"
        cls.families = {
            row["family_id"]: row
            for split in ("train", "dev", "test")
            for row in read_jsonl(cls.data / f"families_{split}.jsonl")
        }

    def test_source_split_has_no_leakage(self) -> None:
        manifest = json.loads((self.data / "split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["assignment"]), 149)
        groups = {name: {key for key, value in manifest["assignment"].items() if value == name} for name in ("train", "dev", "test")}
        self.assertFalse(groups["train"] & groups["dev"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["dev"] & groups["test"])

    def test_all_family_trajectories_are_trainable(self) -> None:
        observed = set()
        for split in ("train", "dev", "test"):
            for dialogue in read_jsonl(self.data / f"sft_{split}.jsonl"):
                metadata = dialogue["metadata"]
                if metadata["state"] == "FULL":
                    continue
                record = self.families[metadata["family_id"]]
                messages = dialogue["messages"]
                first = parse_response(messages[2]["content"])
                final = parse_response(messages[4]["content"])
                expected = "REQUEST_INFORMATION" if metadata["state"] == "MISSING" else "REQUEST_CONFIRMATION"
                self.assertEqual(first["action"], expected)
                self.assertTrue(first["clarification"])
                self.assertEqual(final["action"], "ANSWER")
                self.assertTrue(answer_is_correct(final["answer"], record))
                observed.add(metadata["family_id"])
        self.assertEqual(len(observed), 200)

    def test_ablation_views_exist(self) -> None:
        for split in ("train", "dev", "test"):
            self.assertTrue((self.data / f"sft_initial_only_{split}.jsonl").is_file())
            self.assertTrue((self.data / f"sft_full_only_{split}.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()

