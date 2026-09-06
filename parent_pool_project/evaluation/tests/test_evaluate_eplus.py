from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import evaluate_eplus as evaluator


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        response = {
            "id": "test",
            "model": request["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Answer: Paris"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class EvaluatorTests(unittest.TestCase):
    def test_default_model_is_flash(self) -> None:
        argv = [
            "evaluate_eplus.py",
            "--input", "input.jsonl",
            "--config", "models.json",
            "--output-root", "results",
        ]
        with patch.object(sys, "argv", argv):
            args = evaluator.parse_args()
        self.assertEqual(args.models, ["deepseek_v4_flash"])

    def test_supporting_only_context_is_default(self) -> None:
        sample = {
            "样本ID": "sample-1",
            "完整上下文": [
                {"文档ID": "support", "标题": "Gold", "文本": "Answer evidence."},
                {"文档ID": "noise", "标题": "Noise", "文本": "Distractor text."},
            ],
            "关键证据单元": [{"文档ID": "support", "文本": "Answer evidence."}],
        }
        rendered = evaluator.render_context(sample)
        self.assertIn("Answer evidence.", rendered)
        self.assertNotIn("Distractor text.", rendered)
        full = evaluator.render_context(sample, "full")
        self.assertIn("Distractor text.", full)

    def test_missing_support_document_fails_closed(self) -> None:
        sample = {
            "样本ID": "sample-2",
            "完整上下文": [{"文档ID": "doc-1", "文本": "Text."}],
            "关键证据单元": [{"文档ID": "missing", "文本": "Gold."}],
        }
        with self.assertRaises(ValueError):
            evaluator.render_context(sample)

    def test_normalization_and_scoring(self) -> None:
        self.assertEqual(evaluator.normalize_answer("The New-York."), "new york")
        self.assertEqual(evaluator.normalize_answer("15,848"), "15848")
        score = evaluator.score_prediction(
            "Fernie Alpine Resort.",
            ["Fernie Alpine Resort"],
        )
        self.assertEqual(score["标准答案EM"], 1)
        self.assertEqual(score["别名匹配"], 1)
        self.assertEqual(score["最佳TokenF1"], 1.0)

    def test_alias_and_partial_f1(self) -> None:
        score = evaluator.score_prediction(
            "Mike Stoller",
            ["Jerry Leiber and Mike Stoller", "Mike Stoller"],
        )
        self.assertEqual(score["标准答案EM"], 0)
        self.assertEqual(score["别名匹配"], 1)
        self.assertEqual(score["最佳TokenF1"], 1.0)

    def test_extract_prediction(self) -> None:
        self.assertEqual(evaluator.extract_prediction("Answer: Paris\n"), "Paris")
        self.assertEqual(evaluator.extract_prediction('{"answer":"Warsaw"}'), "Warsaw")
        fence = chr(96) * 3
        self.assertEqual(
            evaluator.extract_prediction(
                f'{fence}json\n{{"answer":"Rome"}}\n{fence}'
            ),
            "Rome",
        )

    def test_deterministic_sample(self) -> None:
        rows = [
            {"来源": "hotpotqa", "样本ID": str(index)}
            for index in range(20)
        ]
        first = evaluator.select_samples(rows, "all", 5, 123)
        second = evaluator.select_samples(rows, "all", 5, 123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_openai_compatible_adapter(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        runtime = evaluator.RuntimeModel(
            alias="local_test",
            provider="vllm_local",
            model_id="test-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key_env="VLLM_API_KEY",
            api_key="EMPTY",
            workers=1,
            timeout_seconds=5,
            max_retries=0,
            request_body={"temperature": 0, "max_tokens": 16},
        )
        try:
            payload, latency = evaluator.call_chat(
                runtime,
                [{"role": "user", "content": "Capital of France?"}],
            )
        finally:
            server.shutdown()
            server.server_close()
        self.assertGreaterEqual(latency, 0)
        self.assertEqual(evaluator.response_content(payload), "Answer: Paris")
        self.assertEqual(evaluator.usage_fields(payload)["总tokens"], 11)

    def test_vllm_key_is_optional(self) -> None:
        config = {
            "默认参数": {
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 16,
                "timeout_seconds": 5,
                "max_retries": 0,
            },
            "服务商": {
                "vllm_local": {
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key_env": "VLLM_API_KEY",
                    "api_key_optional": True,
                }
            },
            "模型": {
                "qwen3_8b": {
                    "服务商": "vllm_local",
                    "model_id": "qwen3-8b",
                    "workers": 1,
                    "extra_body": {},
                }
            },
        }
        args = Namespace(
            provider=None,
            api_key_env=None,
            model_id=None,
            base_url=None,
            workers=None,
        )
        previous = os.environ.pop("VLLM_API_KEY", None)
        try:
            runtime = evaluator.resolve_runtime(config, "qwen3_8b", args)
        finally:
            if previous is not None:
                os.environ["VLLM_API_KEY"] = previous
        self.assertEqual(runtime.api_key, "EMPTY")

    def test_resume_keeps_latest_and_retries_errors(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [
                {"样本ID": "a", "状态": "错误"},
                {"样本ID": "b", "状态": "成功"},
                {"样本ID": "a", "状态": "成功"},
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            latest = evaluator.latest_rows(path)
            self.assertEqual(len(latest), 2)
            self.assertEqual({row["状态"] for row in latest}, {"成功"})
            self.assertEqual(evaluator.completed_ids(path), {"a", "b"})

    def test_resume_does_not_mix_context_modes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [
                {
                    "样本ID": "legacy",
                    "状态": "成功",
                    "提示版本": "closed_context_answer_only.v1",
                },
                {
                    "样本ID": "clean",
                    "状态": "成功",
                    "提示版本": evaluator.PROMPT_VERSION,
                    "上下文模式": "supporting_only",
                },
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            completed = evaluator.completed_ids(
                path,
                evaluator.PROMPT_VERSION,
                "supporting_only",
            )
            self.assertEqual(completed, {"clean"})

    def test_local_secrets_file_does_not_override_environment(self) -> None:
        previous_deepseek = os.environ.get("DEEPSEEK_API_KEY")
        previous_dashscope = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "existing"
        os.environ.pop("DASHSCOPE_API_KEY", None)
        try:
            with TemporaryDirectory() as directory:
                path = Path(directory) / ".env.local"
                path.write_text(
                    "DEEPSEEK_API_KEY=replacement\nDASHSCOPE_API_KEY=loaded\n",
                    encoding="utf-8",
                )
                loaded = evaluator.load_secrets_file(path)
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "existing")
            self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "loaded")
            self.assertEqual(loaded, {"DASHSCOPE_API_KEY"})
        finally:
            if previous_deepseek is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = previous_deepseek
            if previous_dashscope is None:
                os.environ.pop("DASHSCOPE_API_KEY", None)
            else:
                os.environ["DASHSCOPE_API_KEY"] = previous_dashscope


if __name__ == "__main__":
    unittest.main()
