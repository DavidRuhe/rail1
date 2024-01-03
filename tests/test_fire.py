import unittest
from unittest.mock import patch, MagicMock
import rail1.fire


class TestAddSweepName(unittest.TestCase):
    @patch(
        "rail1.fire.os.environ",
        {
            "WANDB_SWEEP_ID": "123",
            "WANDB_PROJECT": "test_project",
            "WANDB_ENTITY": "test_entity",
        },
    )
    @patch("rail1.fire.wandb.Api")
    def test__add_sweep_name(self, mock_wandb_api):
        # Mock the Api method and the sweep object
        mock_sweep = MagicMock()
        mock_sweep.config = {"name": "sweep_name"}
        mock_wandb_api.return_value.sweep.return_value = mock_sweep

        # Call the function with a mock name
        result = rail1.fire._add_sweep_name("test_name")

        # Check if the result is as expected
        self.assertEqual(result, "sweep_name_test_name")

    @patch("rail1.fire.os.environ", {"RANK": "0", "LOCAL_RANK": "1", "WORLD_SIZE": "2"})
    @patch("rail1.fire.dist.init_process_group")
    def test__setup_torchelastic(self, mock_init_process_group):
        # Call the function
        rank, local_rank, world_size = rail1.fire._setup_torchelastic()

        # Check if the function returns the correct values
        self.assertEqual(rank, 0)
        self.assertEqual(local_rank, 1)
        self.assertEqual(world_size, 2)

        # Check if init_process_group was called with the correct arguments
        mock_init_process_group.assert_called_once_with(
            backend="nccl", init_method="env://"
        )

    @patch(
        "rail1.fire.os.environ",
        {
            "SLURM_PROCID": "2",
            "SLURM_NODEID": "1",
            "SLURM_LOCALID": "0",
            "SLURM_NTASKS": "4",
            "NCCL_SYNC_FILE": "/path/to/sync_file",
        },
    )
    @patch("rail1.fire.dist.init_process_group")
    def test_setup_slurm(self, mock_init_process_group):
        # Call the function
        rank, local_rank, world_size = rail1.fire._setup_slurm()

        # Check if the function returns the correct values
        self.assertEqual(rank, 2)  # Adjust this based on the expected calculation
        self.assertEqual(local_rank, 0)
        self.assertEqual(world_size, 4)

        # Check if init_process_group was called with the correct arguments
        mock_init_process_group.assert_called_once_with(
            backend="nccl",
            init_method="file:///path/to/sync_file",
            world_size=4,
            rank=2,
        )

    @patch("rail1.fire.socket.gethostname", return_value="test_hostname")
    @patch("rail1.fire.dist.is_initialized", return_value=True)
    @patch("rail1.fire._setup_torchelastic", return_value=(0, 0, 2))
    @patch(
        "rail1.fire.os.environ",
        {"CUDA_VISIBLE_DEVICES": "0,1", "TORCHELASTIC_RUN_ID": "123"},
    )
    def test_joe(
        self,
        mock_setup_torchelastic,
        mock_is_initialized,
        mock_gethostname,
    ):
        # Call the function
        result = rail1.fire._ddp_setup()

        # Check if the function returns the correct values
        self.assertEqual(
            result, {"rank": 0, "local_rank": 0, "world_size": 2, "device": "cuda:0"}
        )

        # Additional checks
        mock_setup_torchelastic.assert_called_once()
        self.assertTrue(mock_is_initialized())

    @patch("rail1.fire.os.environ", {"WANDB_SWEEP_ID": "test_sweep_id"})
    @patch("rail1.fire.subprocess.getoutput")
    @patch("rail1.fire.dist.is_initialized", return_value=True)
    @patch("rail1.fire.dist.get_rank", return_value=0)
    @patch("rail1.fire.wandb.init")
    def test_setup_wandb(
        self,
        mock_wandb_init,
        mock_get_rank,
        mock_is_initialized,
        mock_getoutput,
    ):
        # Mock subprocess.getoutput to return a matching commit hash and tag
        mock_getoutput.side_effect = ["test_commit_hash", "test_sweep_id"]

        # Call the function
        result = rail1.fire._setup_wandb()

        # Check if wandb.init is called
        mock_wandb_init.assert_called_once()

    @patch("rail1.fire.os.environ", {"WANDB_SWEEP_ID": "test_sweep_id"})
    @patch("rail1.fire.subprocess.getoutput")
    @patch("rail1.fire.dist.is_initialized", return_value=True)
    @patch("rail1.fire.dist.get_rank", return_value=0)
    def test_setup_wandb_error(
        self, mock_get_rank, mock_is_initialized, mock_getoutput
    ):
        # Mock subprocess.getoutput to return a non-matching commit hash and tag
        mock_getoutput.side_effect = ["test_commit_hash", "different_tag"]

        # Call the function and expect a RuntimeError
        with self.assertRaises(RuntimeError):
            rail1.fire._setup_wandb()

    @patch("rail1.fire.argparse.parse_args")
    def test_fire(self, mock_parse_args):
        mock_parse_args.return_value = {"seed": 42, "deterministic": False}

        # Mock function to be passed to fire
        mock_function = MagicMock()

        # Call the fire function
        rail1.fire.fire(mock_function)

        # Check if the mock function was called with the expected config
        expected_config = {
            "seed": 42,
            "deterministic": False,
            "dist": None,
            "wandb": None,
        }
        mock_function.assert_called_once_with(expected_config)


if __name__ == "__main__":
    unittest.main()
