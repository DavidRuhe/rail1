import unittest
import unittest.mock
from collections import defaultdict
import datetime

import torch
from torch import nn, optim

from rail1.training import fit

from torch.utils.data import DataLoader, RandomSampler


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
        self.device = torch.device("cuda:0")  # Example device

    def mock_tensor(self, requires_grad=True):
        tensor = unittest.mock.MagicMock(spec=torch.Tensor)
        tensor.requires_grad = requires_grad
        tensor.to.return_value = tensor
        tensor.detach.return_value = tensor
        return tensor

    def test_to_device_tensor(self):
        tensor = self.mock_tensor()
        result = fit.to_device(tensor, self.device)
        tensor.to.assert_called_with(self.device)
        tensor.detach.assert_called()
        self.assertIs(result, tensor)

    def test_to_device_tensor_no_detach(self):
        tensor = self.mock_tensor()
        result = fit.to_device(tensor, self.device, detach=False)
        tensor.to.assert_called_with(self.device)
        tensor.detach.assert_not_called()
        self.assertIs(result, tensor)

    def test_to_device_list(self):
        tensor_list = [self.mock_tensor(), self.mock_tensor()]
        result = fit.to_device(tensor_list, self.device)
        for tensor in tensor_list:
            tensor.to.assert_called_with(self.device)
            tensor.detach.assert_called()
        self.assertEqual(result, tensor_list)

    def test_to_device_tuple(self):
        tensor_tuple = (self.mock_tensor(), self.mock_tensor())
        result = fit.to_device(tensor_tuple, self.device)
        for tensor in result:  # result should be a list now
            tensor.to.assert_called_with(self.device)
            tensor.detach.assert_called()
        self.assertIsInstance(result, list)

    def test_to_device_dict(self):
        tensor_dict = {"a": self.mock_tensor(), "b": self.mock_tensor()}
        result = fit.to_device(tensor_dict, self.device)
        for key in tensor_dict:
            tensor_dict[key].to.assert_called_with(self.device)
            tensor_dict[key].detach.assert_called()
        self.assertEqual(result, tensor_dict)


class TestApplyMetricFns(unittest.TestCase):
    def test_apply_metric_fns(self):
        # Mock metric functions
        metric_fn1 = unittest.mock.MagicMock(return_value={"metric1": 10})
        metric_fn2 = unittest.mock.MagicMock(return_value={"metric2": 20})

        # Metric dictionaries to be passed to the functions
        metric_dicts = {"data": "example"}

        # Apply metric functions
        result = fit.apply_metric_fns(metric_dicts, [metric_fn1, metric_fn2])

        # Check if the functions were called with the right arguments
        metric_fn1.assert_called_with(metric_dicts)
        metric_fn2.assert_called_with(metric_dicts)

        # Check if the results are aggregated correctly
        self.assertEqual(result, {"metric1": 10, "metric2": 20})


class TestAppendToMetrics(unittest.TestCase):
    def mock_tensor(self, device="cpu"):
        tensor = unittest.mock.MagicMock(spec=["to", "detach"])
        tensor.to.return_value = tensor
        tensor.detach.return_value = tensor
        return tensor

    @unittest.mock.patch("rail1.training.fit.to_device")
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

    @unittest.mock.patch("rail1.training.fit.to_device")
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


class TestLoopTestCase(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_train_state = {
            "limit_val_batches": 4,
            "device": "cpu",
            "global_step": 1,
        }
        self.mock_model = unittest.mock.Mock()
        self.mock_forward_and_loss_fn = unittest.mock.Mock(
            return_value=(None, {"loss": torch.tensor(0.5)})
        )
        self.mock_test_loader = [
            (torch.tensor([0.0]), torch.tensor([1.0])) for _ in range(5)
        ]  # Assume 5 batches
        self.mock_metric_fns = [
            unittest.mock.Mock(return_value={"metric1": 1}),
            unittest.mock.Mock(return_value={"metric2": 2}),
        ]
        self.mock_log_metrics_fn = unittest.mock.Mock()

    @unittest.mock.patch(
        "rail1.training.fit.time.time", side_effect=[100, 110]
    )  # Mock time
    def test_test_loop_validation(self, mock_time):
        # Test the function in validation mode
        fit.test_loop(
            self.mock_train_state,
            self.mock_model,
            self.mock_forward_and_loss_fn,
            self.mock_test_loader,
            self.mock_metric_fns,
            self.mock_log_metrics_fn,
            validation=True,
        )

        # Assertions to verify expected behavior
        self.mock_model.eval.assert_called()
        self.mock_log_metrics_fn.assert_called()

    @unittest.mock.patch(
        "rail1.training.fit.time.time", side_effect=[100, 110]
    )  # Mock time
    def test_test_loop_testing(self, mock_time):
        # Test the function in non-validation mode
        fit.test_loop(
            self.mock_train_state,
            self.mock_model,
            self.mock_forward_and_loss_fn,
            self.mock_test_loader,
            self.mock_metric_fns,
            self.mock_log_metrics_fn,
            validation=False,
        )

        # Assertions to verify expected behavior
        self.mock_model.eval.assert_called()
        self.mock_log_metrics_fn.assert_called()


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
            loss = outputs.mean()  # Simplified loss for testing
            return loss, {"outputs": outputs}

        fit.train_step(
            self.train_state,
            self.model,
            self.optimizer,
            forward_and_loss_fn,
            batch,
            self.print_interval,
        )
        self.assertIsNone(self.train_state["should_raise"])

        def forward_and_loss_fn(batch, model):
            outputs = model(batch)
            return outputs.sum() / 0, {"outputs": outputs}

        fit.train_step(
            self.train_state,
            self.model,
            self.optimizer,
            forward_and_loss_fn,
            batch,
            self.print_interval,
        )

        self.assertTrue(self.train_state["should_raise"] is not None)


class TestFitFunction(unittest.TestCase):
    def setUp(self):
        self.mock_model = unittest.mock.Mock()
        mock_param = unittest.mock.Mock(spec=torch.nn.Parameter)
        mock_param.device = torch.device("cpu")  # Set the device attribute as needed
        self.mock_model.parameters.return_value = iter([mock_param])

        # Mock an optimizer with param_groups
        self.mock_optimizer = unittest.mock.Mock()
        self.mock_optimizer.param_groups = [{"lr": 0.001}]

        # Mock a DataLoader with a RandomSampler
        self.mock_train_loader = unittest.mock.MagicMock(spec=DataLoader)
        self.mock_train_loader.sampler = RandomSampler([3])
        self.mock_train_loader.__iter__.return_value = iter(
            [(torch.tensor([0.0]), torch.tensor([1.0])) for _ in range(5)]
        )
        self.mock_train_loader.__len__.return_value = 5

        self.mock_data = {
            "train_loader": self.mock_train_loader,
            "val_loader": self.mock_train_loader,
        }

        # Mock metric functions as iterables
        self.mock_train_metric_fn = unittest.mock.Mock()
        self.mock_train_metric_fn.return_value = {"train_metric": 1}
        self.mock_eval_metric_fn = unittest.mock.Mock()
        self.mock_eval_metric_fn.return_value = {"eval_metric": 1}
        self.mock_train_metrics_fns = [self.mock_train_metric_fn]
        self.mock_eval_metrics_fns = [self.mock_eval_metric_fn]

        self.mock_log_metrics_fn = unittest.mock.Mock()
        mock_loss = torch.tensor(0.5, requires_grad=True)
        self.mock_forward_and_loss_fn = unittest.mock.Mock(return_value=(mock_loss, {}))

    def test_fit(self):
        fit.fit(
            self.mock_model,
            self.mock_optimizer,
            self.mock_data,
            self.mock_forward_and_loss_fn,
            self.mock_train_metrics_fns,
            self.mock_eval_metrics_fns,
            self.mock_log_metrics_fn,
        )


class TestShouldStopFunction(unittest.TestCase):
    def test_should_stop_conditions(self):
        current_time = datetime.datetime(2022, 1, 1, 12, 0, 0)
        starting_time = datetime.datetime(2022, 1, 1, 10, 0, 0)

        # Case 1: Should stop by max_time
        state = {
            "max_time": datetime.timedelta(hours=1),
            "starting_time": starting_time,
            "max_steps": None,
            "global_step": 0,
        }
        self.assertTrue(fit.should_stop(state), "Failed to stop by max_time")

        # Case 2: Should stop by max_steps
        state["max_time"] = None
        state["max_steps"] = 10
        state["global_step"] = 10
        self.assertTrue(fit.should_stop(state), "Failed to stop by max_steps")

        # Case 3: Should not stop
        state["max_time"] = datetime.timedelta(hours=2)
        state["global_step"] = 5
        fit.should_stop(state)


if __name__ == "__main__":
    unittest.main()
