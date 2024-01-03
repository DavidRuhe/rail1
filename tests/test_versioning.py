import unittest
from unittest import mock
from rail1.utils import versioning


class TestGitDetached(unittest.TestCase):
    @mock.patch("rail1.run.run.subprocess.getoutput")
    def test_git_detached(self, mock_getoutput):
        # Test case 1: HEAD is detached
        mock_getoutput.return_value = "HEAD detached at 1234abc"
        self.assertTrue(versioning.git_detached())

        # Test case 2: HEAD is not detached
        mock_getoutput.return_value = "On branch main"
        self.assertFalse(versioning.git_detached())
