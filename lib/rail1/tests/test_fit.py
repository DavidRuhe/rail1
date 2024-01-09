import datetime
import unittest
import unittest.mock
from collections import defaultdict

import torch
from torch import nn, optim

from rail1 import fit
from rail1.data import batchloader


class TestCountParameters(unittest.TestCase):
    def test_count_parameters(self):
        # Create a mock nn.Module
        mock_module = unittest.mock.MagicMock(spec=nn.Module)

        # Mock the parameters method to return a list of mock parameters
        mock_param1 = unittest.mock.MagicMock(spec=nn.Parameter)
        mock_param1.requires_grad = True
        mock_param1.numel.return_value = 10

        mock_param2 = unittest.mock.MagicMock(spec=nn.Parameter)
        mock_param2.requires_grad = False
        mock_param2.numel.return_value = 20

        mock_module.parameters.return_value = [mock_param1, mock_param2]

        # Call count_parameters with the mock module
        result = fit.count_parameters(mock_module)

        # Assert the result is as expected
        self.assertEqual(result, 10)  # Only mock_param1 is counted


class TestToDevice(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU was detected")
    def test_to_device(self):
        # Case 1: input is a tensor
        x = torch.tensor([1.0, 2.0, 3.0])
        result = fit.to_device(x, self.device)
        self.assertTrue(result.device == self.device)  # type: ignore

        # Case 2: input is a tuple
        x = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]))
        result = fit.to_device(x, self.device)
        self.assertTrue(result[0].device == self.device)  # type: ignore

        # Case 3: input is a dict
        x = {"a": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([4.0, 5.0, 6.0])}
        result = fit.to_device(x, self.device)
        self.assertTrue(result["a"].device == self.device)  # type: ignore


class TestApplyMetricFns(unittest.TestCase):
    def test_apply_metric_fns(self):
        def mean_loss_fn(metrics_dict, is_training):
            del is_training
            return {"loss": metrics_dict["loss"].mean()}
        fit.apply_metric_fns({"loss": torch.randn(32)}, [mean_loss_fn], is_training=True)


class TestAppendToMetrics(unittest.TestCase):
    def mock_tensor(self, device="cpu"):
        tensor = unittest.mock.MagicMock(spec=["to", "detach"])
        tensor.to.return_value = tensor
        tensor.detach.return_value = tensor
        return tensor

    @unittest.mock.patch("rail1.fit.to_device")
    def test_append_to_metrics_without_cpu(self, mock_to_device):
        mock_to_device.side_effect = (
            lambda x, device: x
        )  # Mock to_device to return the input tensor
        result = {"metric1": self.mock_tensor(), "metric2": self.mock_tensor()}
        metrics_dict = {"metric1": [], "metric2": []}

        fit.append_to_metrics_(result, metrics_dict, to_cpu=False)

        for key in result:
            self.assertEqual(len(metrics_dict[key]), 1)
            self.assertIs(metrics_dict[key][0], result[key])

    @unittest.mock.patch("rail1.fit.to_device")
    def test_append_to_metrics_with_cpu(self, mock_to_device):
        mock_to_device.side_effect = (
            lambda x, device: x.detach() if device == "cpu" else x
        )
        result = {"metric1": self.mock_tensor(), "metric2": self.mock_tensor()}
        metrics_dict = {"metric1": [], "metric2": []}

        fit.append_to_metrics_(result, metrics_dict, to_cpu=True)

        for key in result:
            self.assertEqual(len(metrics_dict[key]), 1)
            result[key].detach.assert_called()


# class TestLoopTestCase(unittest.TestCase):
#     def setUp(self):
#         # Mock dependencies
#         self.mock_train_state = {
#             "limit_val_batches": 4,
#             "device": "cpu",
#             "global_step": 1,
#         }
#         self.mock_model = unittest.mock.Mock()
#         self.mock_forward_and_loss_fn = unittest.mock.Mock(
#             return_value=(None, {"loss": torch.tensor(0.5)})
#         )
#         self.mock_test_loader = [
#             (torch.tensor([0.0]), torch.tensor([1.0])) for _ in range(5)
#         ]  # Assume 5 batches
#         self.mock_metric_fns = [
#             unittest.mock.Mock(return_value={"metric1": 1}),
#             unittest.mock.Mock(return_value={"metric2": 2}),
#         ]
#         self.mock_log_metrics_fn = unittest.mock.Mock()

#     @unittest.mock.patch(
#         "rail1.training.fit.time.time", side_effect=[100, 110]
#     )  # Mock time
#     def test_test_loop_validation(self, mock_time):
#         # Test the function in validation mode
#         fit.test_loop(
#             self.mock_train_state,
#             self.mock_model,
#             self.mock_forward_and_loss_fn,
#             self.mock_test_loader,
#             self.mock_metric_fns,
#             self.mock_log_metrics_fn,
#             validation=True,
#         )

#         # Assertions to verify expected behavior
#         self.mock_model.eval.assert_called()
#         self.mock_log_metrics_fn.assert_called()

#     @unittest.mock.patch(
#         "rail1.training.fit.time.time", side_effect=[100, 110]
#     )  # Mock time
#     def test_test_loop_testing(self, mock_time):
#         # Test the function in non-validation mode
#         fit.test_loop(
#             self.mock_train_state,
#             self.mock_model,
#             self.mock_forward_and_loss_fn,
#             self.mock_test_loader,
#             self.mock_metric_fns,
#             self.mock_log_metrics_fn,
#             validation=False,
#         )

#         # Assertions to verify expected behavior
#         self.mock_model.eval.assert_called()
#         self.mock_log_metrics_fn.assert_called()


class TestTrainStep(unittest.TestCase):
    def setUp(self):
        # Setting up a simple model, optimizer, and train state
        self.model = nn.Linear(10, 2)  # simple linear model for testing
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_state = {
            "device": "cpu",
            "should_raise": None,
            "train_metrics": defaultdict(list),
            "global_step": 0,
        }
        self.print_interval = 32

    def test_train_step(self):
        # Creating a dummy batch of data
        batch = torch.randn(5, 10)  # 5 samples, 10 features each

        # Running the train_step function

        def forward_and_loss_fn(batch, model):
            outputs = model(batch)
            loss = outputs.mean(-1)  # Simplified loss for testing
            return loss.mean(), {"outputs": outputs}

        fit.train_step(
            self.train_state,
            self.model,
            self.optimizer,
            forward_and_loss_fn,
            batch,
            self.print_interval,
        )
        self.assertIsNone(self.train_state["should_raise"])

        def forward_and_loss_fn_inf(batch, model):
            outputs = model(batch)
            return outputs.sum() / 0, {"outputs": outputs.mean(-1)}

        fit.train_step(
            self.train_state,
            self.model,
            self.optimizer,
            forward_and_loss_fn_inf,
            batch,
            self.print_interval,
        )

        self.assertTrue(self.train_state["should_raise"] is not None)


class TestFit(unittest.TestCase):
    def setUp(self):
        # Setting up a simple model, optimizer, and train state
        self.model = nn.Linear(10, 2)  # simple linear model for testing
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.run_dir = "tests/runs/test_fit"
        dataset = torch.randn(100, 10)
        self.train_loader = batchloader.BatchLoader(
            dataset, batch_size=10, shuffle=True, collate_fn=lambda x: torch.stack(x)
        )
        self.test_loader = batchloader.BatchLoader(
            dataset, batch_size=10, shuffle=False, collate_fn=lambda x: torch.stack(x)
        )
        self.val_loader = self.test_loader
        self.datasets = {
            "train_loader": self.train_loader,
            "test_loader": self.test_loader,
            "val_loader": self.val_loader,
        }

        def forward_and_loss_fn(batch, model):
            outputs = model(batch)
            loss = outputs.mean(dim=-1)
            return loss.mean(0), {"loss": loss}

        self.forward_and_loss_fn = forward_and_loss_fn

        self.metric_fns = []
        self.log_metrics_fn = lambda x, step: print(x)

    def test_fit(self):

        with self.assertRaises(NotImplementedError):
            fit.fit(
                self.run_dir,
                self.model,
                self.optimizer,
                self.datasets,
                self.forward_and_loss_fn,
                self.metric_fns,
                self.log_metrics_fn,
                max_steps=1,
                scheduler=1
            )


        
        # Running the fit function
        result = fit.fit(
            self.run_dir,
            self.model,
            self.optimizer,
            self.datasets,
            self.forward_and_loss_fn,
            self.metric_fns,
            self.log_metrics_fn,
            max_steps=1,
        )

        self.assertTrue(result)

    def test_should_stop(self):

        starttime = datetime.datetime.now() - datetime.timedelta(minutes=10)
        state = {"starting_time": starttime}
        self.assertTrue(fit.should_stop(state, max_time=datetime.timedelta(minutes=5)))

        state = {"global_step": 100}
        self.assertTrue(fit.should_stop(state, max_steps=10))






if __name__ == "__main__":
    unittest.main()
