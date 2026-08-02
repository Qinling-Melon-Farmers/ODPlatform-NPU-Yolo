import unittest

from od_platform.agent.templates import (
    TEMPLATES,
    get_template,
    list_templates,
    template_instruction,
)


class TestTaskTemplates(unittest.TestCase):
    def test_three_required_templates(self) -> None:
        names = list_templates()
        self.assertEqual(
            set(names),
            {"prepare_dataset", "train_and_evaluate", "annotate_and_review"},
        )

    def test_template_steps_present(self) -> None:
        template = get_template("train_and_evaluate")
        self.assertIsNotNone(template)
        if template is not None:
            self.assertTrue(any("odp-train" in step for step in template.steps))
            self.assertTrue(any("odp-val" in step for step in template.steps))

    def test_template_instruction_contains_steps(self) -> None:
        instruction = template_instruction("prepare_dataset")
        self.assertIn("odp-import-dataset", instruction)
        self.assertIn("odp-validate", instruction)
        self.assertIn("prepare_dataset", instruction)

    def test_unknown_template_returns_empty(self) -> None:
        self.assertEqual(template_instruction("no-such-template"), "")
        self.assertIsNone(get_template("no-such-template"))

    def test_all_templates_have_description(self) -> None:
        for name, template in TEMPLATES.items():
            self.assertTrue(template.description, f"模板 {name} 缺描述")
            self.assertTrue(template.steps, f"模板 {name} 缺步骤")


if __name__ == "__main__":
    unittest.main()
