import random
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from rail1.utils import (
    bash,
    load_module,
    math,
    path,
    printing,
    seed,
    versioning,
)


class TestPath(unittest.TestCase):
    @patch("rail1.utils.path.os.walk")
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
        result_files = list(path.rglob("/test", "*.txt"))
        self.assertListEqual(result_files, expected_files)

        # Test with skip patterns
        expected_files_skipped = ["/test/file1.txt", "/test/dir2/file5.txt"]
        result_files_skipped = list(
            path.rglob("/test", "*.txt", skip_patterns=["dir1"])
        )
        self.assertListEqual(result_files_skipped, expected_files_skipped)

    def test_split_path(self):
        # Test case with a specific file path and split count
        file_path = "/a/b/c/d/e/file.txt"
        k = 3
        expected_result = "/a/b/c"
        self.assertEqual(path.split_path(file_path, k), expected_result)

        # Test case with root directory
        file_path = "/file.txt"
        k = 1
        expected_result = "/"
        self.assertEqual(path.split_path(file_path, k), expected_result)

        # Test case with more splits than path depth
        file_path = "/a/b"
        k = 5
        expected_result = "/"
        self.assertEqual(path.split_path(file_path, k), expected_result)





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


class TestBash(unittest.TestCase):
    def test_run_bash_command(self):
        command = "echo 'hello world'"
        expected_output = "hello world\n"
        output = bash.run_bash_command(command)
        self.assertEqual(output, expected_output)

        command = "ls non_existent_directory"
        output = bash.run_bash_command(command)
        self.assertTrue("No such file or directory" in output)


class TestMath(unittest.TestCase):
    def test_floordiv(self):
        self.assertEqual(math.floordiv(10, 3), 3)
        self.assertEqual(math.floordiv(-10, 3), -4)
        self.assertEqual(math.floordiv(10, -3), -4)
        self.assertEqual(math.floordiv(-10, -3), 3)
        self.assertEqual(math.floordiv(0, 1), 0)
        self.assertRaises(ZeroDivisionError, math.floordiv, 1, 0)

    def test_ceildiv(self):
        self.assertEqual(math.ceildiv(10, 3), 4)
        self.assertEqual(math.ceildiv(-10, 3), -3)
        self.assertEqual(math.ceildiv(10, -3), -3)
        self.assertEqual(math.ceildiv(-10, -3), 4)
        self.assertEqual(math.ceildiv(0, 1), 0)
        self.assertRaises(ZeroDivisionError, math.ceildiv, 1, 0)


class TestPrinting(unittest.TestCase):
    def test_add_prefix(self):
        self.assertEqual(
            printing.add_prefix({"key1": "value1", "key2": "value2"}, "prefix"),
            {"prefix/key1": "value1", "prefix/key2": "value2"},
        )


class TestSeed(unittest.TestCase):
    def test_set_seed(self):
        seed.set_seed(0)
        random_first = random.random()
        torch_first = torch.rand(1)
        numpy_first = np.random.rand(1)

        seed.set_seed(0)
        random_second = random.random()
        torch_second = torch.rand(1)
        numpy_second = np.random.rand(1)

        self.assertEqual(random_first, random_second)
        self.assertEqual(torch_first, torch_second)
        self.assertEqual(numpy_first, numpy_second)


class TestVersioning(unittest.TestCase):
    def test_git_detached(self):
        versioning.git_detached()

    @patch("subprocess.getoutput")
    def test_git_detached_true(self, mock_getoutput):
        # Set the mock to return a string indicating a detached HEAD
        mock_getoutput.return_value = "HEAD detached at 1234abc"
        with self.assertRaises(RuntimeError) as context:
            versioning.check_git_detached()
        self.assertEqual(
            str(context.exception), "git is a detached HEAD. Please checkout a branch."
        )


if __name__ == "__main__":
    unittest.main()
