from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import resolve_output_artifact  # noqa: E402


class OutputArtifactPathTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.experiment = {"paths": {"output_dir": str(self.root / "baselines")}}

    def test_relative_value_resolves_against_output_dir(self):
        self.assertEqual(
            resolve_output_artifact(self.experiment, "qwen25_sft_train/predictions.jsonl"),
            self.root / "baselines" / "qwen25_sft_train" / "predictions.jsonl",
        )

    def test_absolute_value_is_kept(self):
        absolute = self.root / "elsewhere" / "adapter"
        self.assertEqual(resolve_output_artifact(self.experiment, str(absolute)), absolute)


if __name__ == "__main__":
    unittest.main()
