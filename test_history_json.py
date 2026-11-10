import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "json"
FILES = ("value_iteration_history.json", "policy_iteration_history.json")


def reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


class HistoryJsonTests(unittest.TestCase):
    # Both required history files must exist.
    def test_files_exist(self):
        for filename in FILES:
            with self.subTest(file=filename):
                self.assertTrue((ROOT / filename).is_file(), f"Missing {filename}")

    # Each file must contain valid JSON with numbered 4 × 12 numeric grids.
    def test_json_structure(self):
        for filename in FILES:
            with self.subTest(file=filename):
                history = json.loads(
                    (ROOT / filename).read_text(encoding="utf-8"),
                    parse_constant=reject_constant,
                )
                self.assertIsInstance(history, dict)
                self.assertTrue(history, "History must not be empty")
                for iteration, grid in history.items():
                    with self.subTest(iteration=iteration):
                        self.assertRegex(iteration, r"\A(?:0|[1-9][0-9]*)\Z")
                        self.assertIsInstance(grid, list)
                        self.assertEqual(len(grid), 4)
                        for row in grid:
                            self.assertIsInstance(row, list)
                            self.assertEqual(len(row), 12)
                            for value in row:
                                self.assertIn(type(value), (int, float))

    # Numbers may have at most two decimal places; trailing zeros are optional.
    def test_two_decimal_places(self):
        def check_number(token):
            self.assertRegex(token, r"\A-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?\Z")
            return float(token)

        for filename in FILES:
            with self.subTest(file=filename):
                json.loads(
                    (ROOT / filename).read_text(encoding="utf-8"),
                    parse_int=check_number,
                    parse_float=check_number,
                    parse_constant=reject_constant,
                )


if __name__ == "__main__":
    unittest.main()
