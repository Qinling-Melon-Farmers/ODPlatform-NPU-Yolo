import tempfile
import unittest
from pathlib import Path

from od_platform.cli.import_dataset import main as import_dataset_main
from od_platform.cli.plot_training import main as plot_training_main
from od_platform.cli.transform_data import build_parser as build_transform_parser


class TestCliEntrypoints(unittest.TestCase):
    def test_import_dataset_reports_missing_zip_as_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            import_dataset_main(["missing.zip"])

        self.assertEqual(exc.exception.code, 2)

    def test_plot_training_reports_missing_csv_as_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            plot_training_main(["missing-results.csv"])

        self.assertEqual(exc.exception.code, 2)

    def test_transform_cli_rejects_segment_task(self) -> None:
        parser = build_transform_parser()

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["--dataset", "demo", "--format", "yolo", "--task", "segment"])

        self.assertEqual(exc.exception.code, 2)

    def test_transform_cli_accepts_detect_task(self) -> None:
        parser = build_transform_parser()

        args = parser.parse_args(["--dataset", "demo", "--format", "pascal_voc", "--task", "detect"])

        self.assertEqual(args.task, "detect")

    def test_plot_training_writes_outputs_for_minimal_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "results.csv"
            csv_path.write_text(
                "epoch,train/box_loss,val/box_loss,train/cls_loss,val/cls_loss,"
                "train/dfl_loss,val/dfl_loss,metrics/precision(B),metrics/recall(B),"
                "metrics/mAP50(B),metrics/mAP50-95(B),lr/pg0,lr/pg1,lr/pg2,time\n"
                "1,1,1,1,1,1,1,0.8,0.7,0.6,0.5,0.01,0.01,0.01,2\n",
                encoding="utf-8",
            )
            image_path = root / "plot.png"
            summary_path = root / "summary.json"

            code = plot_training_main([str(csv_path), "--output", str(image_path), "--summary", str(summary_path)])

            self.assertEqual(code, 0)
            self.assertTrue(image_path.exists())
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
