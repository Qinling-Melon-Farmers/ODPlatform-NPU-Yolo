import unittest

from od_platform.agent.schema import build_all_tool_schemas, parser_to_tool_schema
from od_platform.cli.import_dataset import build_parser as build_import_parser
from od_platform.cli.train_model import build_parser as build_train_parser
from od_platform.cli.transform_data import build_parser as build_transform_parser


class TestParserToToolSchema(unittest.TestCase):
    def test_train_tool_schema_mapping(self) -> None:
        schema = parser_to_tool_schema(build_train_parser())
        self.assertIsNotNone(schema)
        if schema is None:
            return
        self.assertEqual(schema.name, "odp-train")
        self.assertIn("epochs", schema.parameters["properties"])
        self.assertEqual(schema.parameters["properties"]["epochs"]["type"], "integer")
        # 控制字段不暴露
        for key in ("config", "executor", "dry_run"):
            self.assertNotIn(key, schema.parameters["properties"])

    def test_store_true_becomes_boolean(self) -> None:
        schema = parser_to_tool_schema(build_import_parser())
        self.assertIsNotNone(schema)
        if schema is None:
            return
        self.assertEqual(schema.parameters["properties"]["overwrite"]["type"], "boolean")

    def test_choices_become_enum(self) -> None:
        schema = parser_to_tool_schema(build_transform_parser())
        self.assertIsNotNone(schema)
        if schema is None:
            return
        format_prop = schema.parameters["properties"]["format"]
        self.assertEqual(format_prop["type"], "string")
        self.assertEqual(set(format_prop["enum"]), {"pascal_voc", "coco", "yolo"})

    def test_positional_marked(self) -> None:
        schema = parser_to_tool_schema(build_import_parser())
        self.assertIsNotNone(schema)
        if schema is None:
            return
        positional = [spec for spec in schema.arg_specs if spec.is_positional]
        self.assertEqual(len(positional), 1)
        self.assertEqual(positional[0].key, "zip_path")
        self.assertIsNone(positional[0].option)

    def test_required_marked(self) -> None:
        schema = parser_to_tool_schema(build_import_parser())
        self.assertIsNotNone(schema)
        if schema is None:
            return
        self.assertIn("zip_path", schema.parameters["required"])

    def test_agent_self_excluded(self) -> None:
        import argparse

        parser = argparse.ArgumentParser(prog="odp-agent", description="AI 助手")
        parser.add_argument("--model")
        self.assertIsNone(parser_to_tool_schema(parser))

    def test_nargs_plus_becomes_array(self) -> None:
        schema = parser_to_tool_schema(build_transform_parser())
        self.assertIsNotNone(schema)
        if schema is None:
            return
        self.assertEqual(schema.parameters["properties"]["classes"]["type"], "array")

    def test_build_all_default_schemas(self) -> None:
        schemas = build_all_tool_schemas()
        names = {schema.name for schema in schemas}
        self.assertIn("odp-train", names)
        self.assertIn("odp-import-dataset", names)
        # 交互式/破坏性工具不暴露
        self.assertNotIn("odp-annotate", names)
        self.assertNotIn("odp-reset", names)
        self.assertNotIn("odp-agent", names)

    def test_schema_is_stable_json_shape(self) -> None:
        schema = parser_to_tool_schema(build_train_parser())
        self.assertIsNotNone(schema)
        if schema is None:
            return
        self.assertEqual(schema.parameters["type"], "object")
        self.assertIn("properties", schema.parameters)


if __name__ == "__main__":
    unittest.main()
