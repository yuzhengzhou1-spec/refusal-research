from __future__ import annotations

import unittest
from pathlib import Path

from common import resolve_output_artifact


class OutputArtifactPathTest(unittest.TestCase):
    def test_relative_value_resolves_against_output_dir(self):
        experiment = {"paths": {"output_dir": str(Path("/srv/experiments/baselines"))}}
        self.assertEqual(
            resolve_output_artifact(experiment, "qwen25_sft_train/predictions.jsonl"),
            Path("/srv/experiments/baselines/qwen25_sft_train/predictions.jsonl"),
        )

    def test_absolute_value_is_kept(self):
        experiment = {"paths": {"output_dir": str(Path("/srv/experiments/baselines"))}}
        self.assertEqual(
            resolve_output_artifact(experiment, "/elsewhere/adapter"),
            Path("/elsewhere/adapter"),
        )


if __name__ == "__main__":
    unittest.main()
