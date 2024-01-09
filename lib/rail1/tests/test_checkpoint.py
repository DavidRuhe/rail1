import shutil
import os
from torch import nn, optim
from unittest.mock import MagicMock, patch
import unittest

from rail1 import checkpoint


class TestCheckpoint(unittest.TestCase):
    @patch("wandb.Artifact")
    @patch("wandb.log_artifact")
    @patch("wandb.run", MagicMock(id="1234"))
    def test_save_wandb(self, mock_log_artifact, mock_artifact_class):
        # Set up the mock Artifact instance
        mock_artifact = MagicMock()
        mock_artifact_class.return_value = mock_artifact

        # Call the function with a test file
        test_file = "test_file.txt"
        checkpoint.save_wandb(test_file)

        # Assert that an Artifact was created with the correct name and type
        mock_artifact_class.assert_called_once_with(
            "1234-checkpoint", type="checkpoint"
        )

        # Assert that the file was added to the artifact
        mock_artifact.add_file.assert_called_once_with(test_file)

        # Assert that the artifact was logged
        mock_log_artifact.assert_called_once_with(mock_artifact)

    def test_save_checkpoint(self):
        # Mock model and optimizer
        mock_model = nn.Linear(1, 1)
        mock_optimizer = optim.Adam(mock_model.parameters())

        # Mock train state
        train_state = {"global_step": 1, "current_epoch": 1}

        # Call the function with test parameters
        checkpoint_dir = "test_dir"
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        checkpoint.save_checkpoint(
            checkpoint_dir,
            mock_model,
            train_state,
            mock_optimizer,
            metrics={"accuracy": 0.99},
        )
        checkpoint.save_checkpoint(
            checkpoint_dir, mock_model, train_state, mock_optimizer, metrics=None
        )

        with patch("torch.distributed.is_initialized", return_value=True):
            with self.assertRaises(NotImplementedError):
                checkpoint.save_checkpoint(
                    checkpoint_dir,
                    mock_model,
                    train_state,
                    mock_optimizer,
                    metrics={"accuracy": 0.99},
                )

        wandb_run = MagicMock()
        wandb_run.id = "1234"
        with patch("wandb.run", wandb_run), patch(
            "wandb.log_artifact", return_value=None
        ):
            checkpoint.save_checkpoint(
                checkpoint_dir,
                mock_model,
                train_state,
                mock_optimizer,
                metrics={"accuracy": 0.99},
            )

    def test_load_checkpoint(self):
        checkpoint_dir = "test_dir"
        model = nn.Linear(1, 1)
        optimizer = optim.Adam(model.parameters())
        train_state = {"global_step": 1, "current_epoch": 1}

        checkpoint.load_checkpoint(
            checkpoint_dir, model, train_state=train_state, optimizer=optimizer
        )
