import unittest
from unittest.mock import patch
from rail1 import argparse


class TestArgparse(unittest.TestCase):
    def test_create_parser_with_empty_dict(self):
        parser = argparse.create_parser({})
        self.assertIsNotNone(parser)

    def test_create_parser_with_nested_dict(self):
        test_config = {"section1": {"param1": 1, "param2": "value"}, "param3": 2.0}
        parser = argparse.create_parser(test_config)
        self.assertIsNotNone(parser)

    @patch(
        "rail1.argparse.sys.argv",
        ["script_name", "config_path", "--section1.param1", "10", "--param3", "3.0"],
    )
    @patch("rail1.utils.load_module.load_attribute_from_python_file")
    def test_parse_args(self, mock_load_attribute):
        mock_load_attribute.return_value = {
            "section1": {"param1": 1, "param2": "value"},
            "param3": 2.0,
            "parameters": {"seed": {"values": [0]}},
        }

        config = argparse.parse_args()
        self.assertEqual(config["section1"]["param1"], 10)
        self.assertEqual(config["param3"], 3.0)


if __name__ == "__main__":
    unittest.main()
