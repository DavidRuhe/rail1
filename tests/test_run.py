import os
import unittest
from rail1.run import run, devrun
from unittest import mock
import tempfile


class TestGenerateSbatchLines(unittest.TestCase):
    def test_single_hyphen_args(self):
        input_str = "-N 1 -p short"
        expected_output = ["#!/bin/bash", "#SBATCH -N 1", "#SBATCH -p short"]
        self.assertEqual(run.generate_sbatch_lines(input_str), expected_output)

    def test_double_hyphen_args(self):
        input_str = "--mem=4G --time=01:00:00"
        expected_output = ["#!/bin/bash", "#SBATCH --mem=4G", "#SBATCH --time=01:00:00"]
        self.assertEqual(run.generate_sbatch_lines(input_str), expected_output)

    def test_double_hyphen_args_separated(self):
        input_str = "--mem 4G --time 01:00:00"
        expected_output = ["#!/bin/bash", "#SBATCH --mem=4G", "#SBATCH --time=01:00:00"]
        self.assertEqual(run.generate_sbatch_lines(input_str), expected_output)

    def test_mixed_args(self):
        input_str = "-N 1 --time=01:00:00 -p short"
        expected_output = [
            "#!/bin/bash",
            "#SBATCH -N 1",
            "#SBATCH --time=01:00:00",
            "#SBATCH -p short",
        ]
        self.assertEqual(run.generate_sbatch_lines(input_str), expected_output)

    def test_empty_input(self):
        input_str = ""
        expected_output = ["#!/bin/bash"]
        self.assertEqual(run.generate_sbatch_lines(input_str), expected_output)

    def test_no_args(self):
        input_str = "--time=01:00:00"
        expected_output = ["#!/bin/bash", "#SBATCH --time=01:00:00"]
        self.assertEqual(run.generate_sbatch_lines(input_str), expected_output)


class TestWriteJobfile(unittest.TestCase):
    def test_write_jobfile(self):
        slurm_string = "--mem=4G --time=01:00:00"
        n_jobs = 5
        command = "echo 'Hello, World!'"
        sweep_id = "123abc"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Call the function with a temporary directory
            run.write_jobfile(slurm_string, n_jobs, command, tmpdir, sweep_id)

            # Construct the expected file path
            file_path = os.path.join(tmpdir, "slurm_job.sh")

            os.listdir(tmpdir)

            # Assert that the file was created
            self.assertTrue(os.path.exists(file_path))

            # Read the file and assert its contents
            with open(file_path, "r") as file:
                contents = file.read()
                self.assertIn(f"#SBATCH --array=1-{n_jobs}", contents)
                self.assertIn(
                    f"#SBATCH --output={os.path.join(tmpdir, 'slurm-%j.out')}", contents
                )
                self.assertIn(f"cd {tmpdir}", contents)
                self.assertIn(f"git checkout {sweep_id}", contents)
                self.assertIn("source ./activate.sh", contents)
                self.assertIn(command, contents)


class TestReplaceVariables(unittest.TestCase):
    def test_replace_variables(self):
        # Test case 1: Single replacement
        command = "echo {name}"
        locals_dict = {"name": "Alice"}
        self.assertEqual(run.replace_variables(command, locals_dict), "echo Alice")

        # Test case 2: Multiple replacements
        command = "copy {src} {dst}"
        locals_dict = {"src": "/source/path", "dst": "/destination/path"}
        self.assertEqual(
            run.replace_variables(command, locals_dict),
            "copy /source/path /destination/path",
        )

        # Test case 3: No replacements
        command = "ls -l"
        locals_dict = {}
        self.assertEqual(run.replace_variables(command, locals_dict), "ls -l")

        # Test case 4: Variable not in locals_dict
        command = "delete {filename}"
        locals_dict = {"file": "document.txt"}
        with self.assertRaises(KeyError):
            run.replace_variables(command, locals_dict)


class TestGitStatus(unittest.TestCase):
    @mock.patch("rail1.run.run.subprocess.run")
    @mock.patch("rail1.run.run.subprocess.getoutput")
    def test_git_status(self, mock_getoutput, mock_run):
        # Test case 1: Diverged
        mock_getoutput.return_value = "1 1"
        self.assertEqual(run.git_status(), "diverged")

        # Test case 2: Ahead
        mock_getoutput.return_value = "1 0"
        self.assertEqual(run.git_status(), "behind")

        # Test case 3: Behind
        mock_getoutput.return_value = "0 1"
        self.assertEqual(run.git_status(), "ahead")

        # Test case 4: Up-to-date
        mock_getoutput.return_value = "0 0"
        self.assertEqual(run.git_status(), "up-to-date")


class TestDevRun(unittest.TestCase):
    @mock.patch("rail1.run.devrun.sys.argv", ["script.py", "not_a_python_file.txt"])
    def test_invalid_file_extension(self):
        """
        Test if the script raises an exception for non-Python config files.
        """
        with self.assertRaises(ValueError):
            devrun.main()

    @mock.patch("rail1.run.devrun.sys.argv", ["script.py", "config.py"])
    @mock.patch("rail1.run.devrun.subprocess")
    @mock.patch("rail1.run.devrun.load_module.load_attribute_from_python_file")
    def test_devrun(self, mock_load_attribute, mock_subprocess):
        """
        Test if the script generates correct command permutations for parameters.
        """
        mock_config = {
            "parameters": {
                "param1": {"values": [1, 2]},
                "param2": {"values": ["a", "b"]},
            },
            "command": ["python", "${program}"],
        }
        mock_load_attribute.return_value = mock_config
        devrun.main()

        mock_config = {
            "parameters": {},
            "command": "python ${program}",
        }
        mock_load_attribute.return_value = mock_config
        devrun.main()


if __name__ == "__main__":
    unittest.main()
