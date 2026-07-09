import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from od_platform.runtime_config.api import build_train_config, build_val_config
from od_platform.runtime_config.generator import (
    ConfigGenerator,
    default_val_config,
    write_val_template,
)
from od_platform.runtime_config.loaders import CLILoader, YAMLLoader, load_val_config
from od_platform.runtime_config.merger import ConfigMerger, ConfigSource
from od_platform.runtime_config.registry import get_config_class, list_config_names
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.runtime_config.val import YOLOValConfig


class TestRuntimeConfig(unittest.TestCase):
    def test_val_config_exports_ultralytics_kwargs_without_framework_fields(self) -> None:
        config = YOLOValConfig(data="steel", model="best.pt", extra_args={"rect": True})

        kwargs = config.to_ultralytics_kwargs()

        self.assertNotIn("model", kwargs)
        self.assertNotIn("task", kwargs)
        self.assertNotIn("extra_args", kwargs)
        self.assertEqual(kwargs["data"], "steel")
        self.assertTrue(kwargs["rect"])

    def test_runtime_configs_reject_segment_until_supported_end_to_end(self) -> None:
        with self.assertRaises(ValueError):
            YOLOTrainConfig(task="segment")

        with self.assertRaises(ValueError):
            YOLOValConfig(task="segment")

    def test_val_template_generator_and_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "val.yaml"

            write_val_template(path)
            loaded = load_val_config(path)

            self.assertIn("split", default_val_config())
            self.assertEqual(loaded.split, "val")

    def test_config_generator_supports_registered_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            output = ConfigGenerator().generate("val", root / "val.yaml")

            self.assertTrue(output.exists())
            self.assertIs(get_config_class("train"), YOLOTrainConfig)
            self.assertEqual(set(list_config_names()), {"train", "val", "infer"})

    def test_yaml_and_cli_loaders_keep_explicit_falsy_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.yaml"
            path.write_text("epochs: 0\nplots: false\nname: ''\nignored:\n", encoding="utf-8")
            namespace = argparse.Namespace(config="train", epochs=0, plots=False, name="", dry_run=True, skipped=None)

            yaml_data = YAMLLoader().load(path)
            cli_data = CLILoader().load(namespace)

            self.assertEqual(yaml_data["epochs"], 0)
            self.assertFalse(yaml_data["plots"])
            self.assertEqual(yaml_data["name"], "")
            self.assertFalse(cli_data["plots"])
            self.assertNotIn("config", cli_data)
            self.assertNotIn("dry_run", cli_data)
            self.assertNotIn("skipped", cli_data)

    def test_config_merger_tracks_override_chain(self) -> None:
        merger = ConfigMerger()

        config = merger.merge(
            YOLOTrainConfig,
            sources=[
                (ConfigSource.YAML, {"epochs": 10, "batch": 16}),
                (ConfigSource.CLI, {"epochs": 2}),
            ],
        )

        self.assertEqual(config.epochs, 2)
        metadata = merger.get_metadata("epochs")
        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata.source, ConfigSource.CLI)
        self.assertEqual([item.source for item in metadata.chain()], [ConfigSource.CLI, ConfigSource.YAML, ConfigSource.DEFAULT])
        self.assertIn("epochs", merger.to_audit_log()["overridden"])

    def test_public_api_builds_with_yaml_and_cli_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs" / "runtime"
            config_dir.mkdir(parents=True)
            (config_dir / "train.yaml").write_text("model: yolo11n.pt\ndata: rsod\nepochs: 10\n", encoding="utf-8")
            args = argparse.Namespace(epochs=3, batch=4, dry_run=True, config="train")

            with patch("od_platform.runtime_config.loaders.paths.RUNTIME_CONFIGS_DIR", config_dir):
                config, merger = build_train_config("train", args)
                preview_config, preview_merger = build_val_config(None, argparse.Namespace(split="test"), dry_run=True)

            self.assertIsNotNone(config)
            assert config is not None
            self.assertEqual(config.epochs, 3)
            self.assertEqual(config.batch, 4)
            self.assertIn("epochs", merger.to_audit_log()["overridden"])
            self.assertIsNone(preview_config)
            self.assertEqual(preview_merger.get_metadata("split").value, "test")


if __name__ == "__main__":
    unittest.main()
