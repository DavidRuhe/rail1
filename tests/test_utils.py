import unittest
from unittest.mock import patch, MagicMock

from rail1.utils import recursive_glob
from rail1.utils import load_module
from rail1.utils import printing


class TestRGlob(unittest.TestCase):
    @patch("rail1.utils.recursive_glob.os.walk")
    def test_rglob(self, mock_walk):
        # Mocking the os.walk to simulate a directory structure
        directory_structure = [
            ("/test", ("dir1", "dir2"), ("file1.txt", "file2.doc")),
            ("/test/dir1", (), ("file3.txt", "file4.doc")),
            ("/test/dir2", (), ("file5.txt", "file6.doc")),
        ]

        mock_walk.return_value = directory_structure

        # Test searching for .txt files
        expected_files = [
            "/test/file1.txt",
            "/test/dir1/file3.txt",
            "/test/dir2/file5.txt",
        ]
        result_files = list(recursive_glob.rglob("/test", "*.txt"))
        self.assertListEqual(result_files, expected_files)

        # Test with skip patterns
        expected_files_skipped = ["/test/file1.txt", "/test/dir2/file5.txt"]
        result_files_skipped = list(recursive_glob.rglob("/test", "*.txt", skip_patterns=["dir1"]))
        self.assertListEqual(result_files_skipped, expected_files_skipped)


class TestLoadAttributeFromPythonFile(unittest.TestCase):
    @patch("rail1.utils.load_module.importlib.util")
    def test_load_attribute_from_python_file(self, mock_importlib_util):
        # Mocking the behavior of importlib.util
        mock_spec = MagicMock()
        mock_importlib_util.spec_from_file_location.return_value = mock_spec
        mock_importlib_util.module_from_spec.return_value = MagicMock()

        # Define a sample attribute to be loaded
        attribute_name = "sample_attribute"
        attribute_value = "Test Value"
        setattr(
            mock_importlib_util.module_from_spec.return_value,
            attribute_name,
            attribute_value,
        )

        # Test loading the attribute
        path = "/path/to/file.py"
        result = load_module.load_attribute_from_python_file(path, attribute_name)
        self.assertEqual(result, attribute_value)

        # Test handling of None spec
        mock_importlib_util.spec_from_file_location.return_value = None
        with self.assertRaises(AssertionError):
            load_module.load_attribute_from_python_file(path, "non_existent_attribute")

    @patch("rail1.utils.load_module.sys.modules", new_callable=dict)
    def test_sys_modules_update(self, mock_sys_modules):
        # Mocking sys.modules
        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_spec.loader.exec_module = MagicMock()

        with patch("rail1.utils.load_module.importlib.util") as mock_importlib_util:
            mock_importlib_util.spec_from_file_location.return_value = mock_spec
            mock_importlib_util.module_from_spec.return_value = MagicMock()

            attribute_name = "sample_attribute"
            path = "/path/to/file.py"
            load_module.load_attribute_from_python_file(path, attribute_name)

            # Check if sys.modules is updated correctly
            self.assertIn(f"{attribute_name}_module", mock_sys_modules)

class TestHumanFormatFloat(unittest.TestCase):

    def test_human_format_float(self):
        # Test the basic functionality
        self.assertEqual(printing.human_format_float(123), "123")
        self.assertEqual(printing.human_format_float(1234), "1.23K")
        self.assertEqual(printing.human_format_float(1234567), "1.23M")
        self.assertEqual(printing.human_format_float(1234567890), "1.23B")

        # Test with negative numbers
        self.assertEqual(printing.human_format_float(-1234), "-1.23K")
        self.assertEqual(printing.human_format_float(-1234567), "-1.23M")

        # Test with zero
        self.assertEqual(printing.human_format_float(0), "0")

        # Test with very small numbers
        self.assertEqual(printing.human_format_float(0.000123), "0.000123")


        # Test precision
        self.assertEqual(printing.human_format_float(999999), "1M")
        self.assertEqual(printing.human_format_float(1000.999), "1K")


if __name__ == "__main__":
    unittest.main()
