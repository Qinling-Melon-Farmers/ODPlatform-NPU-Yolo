import argparse
import importlib
import unittest

from od_platform.agent.schema import ToolSchema
from od_platform.agent.tools import (
    ToolRegistry,
    _argv_from_arguments,
    _redact_arguments,
    build_default_registry,
)


def _mini_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mini", description="迷你测试 CLI")
    parser.add_argument("--epochs", type=int, help="轮数")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--classes", nargs="+", help="类别列表")
    parser.add_argument("--data", help="数据集")
    parser.add_argument("input", help="输入路径")
    return parser


def _mini_schema() -> ToolSchema:
    from od_platform.agent.schema import parser_to_tool_schema

    schema = parser_to_tool_schema(_mini_parser())
    assert schema is not None
    return schema


class TestArgvConstruction(unittest.TestCase):
    def test_scalar_flag_positional(self) -> None:
        schema = _mini_schema()
        argv = _argv_from_arguments(
            schema,
            {"epochs": 3, "verbose": True, "data": "rsod", "input": "a.zip"},
        )
        self.assertIn("--epochs", argv)
        self.assertEqual(argv[argv.index("--epochs") + 1], "3")
        self.assertIn("--verbose", argv)
        self.assertIn("--data", argv)
        self.assertEqual(argv[-1], "a.zip")  # 位置参数在末尾

    def test_false_flag_omitted_and_list_expanded(self) -> None:
        schema = _mini_schema()
        argv = _argv_from_arguments(schema, {"verbose": False, "classes": ["cat", "dog"], "input": "x"})
        self.assertNotIn("--verbose", argv)
        index = argv.index("--classes")
        self.assertEqual(argv[index : index + 3], ["--classes", "cat", "dog"])

    def test_none_values_omitted(self) -> None:
        schema = _mini_schema()
        argv = _argv_from_arguments(schema, {"input": "x"})
        self.assertEqual(argv, ["x"])


class TestRedact(unittest.TestCase):
    def test_redact_masks_api_key(self) -> None:
        redacted = _redact_arguments({"api_key": "sk-123", "dataset": "rsod", "model": "deepseek-chat"})
        self.assertEqual(redacted["api_key"], "***")
        self.assertEqual(redacted["dataset"], "rsod")


class TestMiniCliExecution(unittest.TestCase):
    """用测试内定义的迷你 CLI 验证执行链路（不碰真实平台 CLI）。"""

    def _make_registry(self, *, dry_run: bool = False) -> ToolRegistry:
        import od_platform.tests._agent_mini_cli as mini_cli

        registry = ToolRegistry(dry_run=dry_run)
        registry.register_cli(mini_cli.__name__)
        return registry

    def test_execute_success(self) -> None:
        registry = self._make_registry()
        result = registry.execute("mini", {"input": "ok"})
        self.assertTrue(result.ok)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("成功", result.summary)

    def test_execute_missing_required_returns_failed(self) -> None:
        registry = self._make_registry()
        result = registry.execute("mini", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, 2)  # argparse 缺必填位置参数

    def test_execute_unknown_tool(self) -> None:
        registry = self._make_registry()
        result = registry.execute("no-such-tool", {})
        self.assertFalse(result.ok)
        self.assertIsNone(result.exit_code)

    def test_global_dry_run_appends_flag(self) -> None:
        """dry_run 模式自动追加 --dry-run：训练以计划模式成功（不真实训练）。"""
        real = ToolRegistry(dry_run=True)
        real.register_cli("od_platform.cli.train_model", dry_run_flag=True)
        result = real.execute("odp-train", {})
        self.assertTrue(result.ok)
        self.assertEqual(result.exit_code, 0)

    def test_confirmation_required_refused(self) -> None:
        import od_platform.tests._agent_mini_cli as mini_cli

        registry = ToolRegistry()
        registry.register_cli(mini_cli.__name__, requires_confirmation=True)
        result = registry.execute("mini", {"input": "ok"})
        self.assertFalse(result.ok)
        self.assertIn("确认", result.summary)
        registry.confirm("mini")
        result = registry.execute("mini", {"input": "ok"})
        self.assertTrue(result.ok)

    def test_excluded_args_removed_from_schema(self) -> None:
        try:
            importlib.import_module("od_platform.annotation.cli.auto_annotate")
        except ImportError:
            self.skipTest("odp-auto-annotate 尚未实现（Phase G 后启用）")
        registry = ToolRegistry()
        registry.register_cli("od_platform.annotation.cli.auto_annotate", excluded_args=("--api-key", "--base-url"))
        schema = registry.get_schema("odp-auto-annotate")
        self.assertIsNotNone(schema)
        if schema is not None:
            self.assertNotIn("api_key", schema.parameters["properties"])
            self.assertNotIn("base_url", schema.parameters["properties"])


class TestDefaultRegistry(unittest.TestCase):
    def test_default_registry_contains_core_tools(self) -> None:
        registry = build_default_registry()
        names = registry.names()
        self.assertIn("list_datasets", names)
        self.assertIn("list_available_models", names)
        self.assertIn("odp-list-models", names)
        self.assertIn("odp-train", names)
        self.assertIn("odp-validate", names)
        self.assertNotIn("odp-annotate", names)
        self.assertNotIn("odp-reset", names)

    def test_default_registry_schemas_openai_shape(self) -> None:
        registry = build_default_registry()
        for schema in registry.schemas():
            self.assertEqual(schema["type"], "function")
            self.assertIn("name", schema["function"])
            self.assertIn("parameters", schema["function"])

    def test_service_tools_execute(self) -> None:
        from od_platform.common import paths

        registry = build_default_registry()
        result = registry.execute("list_datasets", {})
        self.assertTrue(result.ok)
        if paths.RAW_DATA_DIR.exists() and any(paths.RAW_DATA_DIR.iterdir()):
            self.assertIn("图片", result.summary)
        else:
            self.assertIn("没有数据集", result.summary)


if __name__ == "__main__":
    unittest.main()
