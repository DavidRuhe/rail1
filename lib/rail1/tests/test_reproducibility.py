import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import unittest
from rail1.utils.seed import set_seed
from rail1 import checkpoint
import subprocess
import numpy as np


class SimpleDataset(data.Dataset):
    def __init__(self):
        self.data = torch.randn(10, 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor([1])


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


class TestReproducibility(unittest.TestCase):
    def test_reproducibility(self):
        set_seed(0, deterministic=True)

        # First run
        dataset1 = SimpleDataset()
        dataloader1 = data.DataLoader(dataset1, batch_size=2)
        model1 = SimpleNet()
        if torch.cuda.is_available():
            model1 = model1.cuda()
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)

        for inputs, _ in dataloader1:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            optimizer1.zero_grad()
            outputs = model1(inputs)
            loss = outputs.mean()
            loss.backward()
            optimizer1.step()

        # Save model state
        state_dict1 = model1.state_dict()

        # Second run with the same seed
        set_seed(0, deterministic=True)

        dataset2 = SimpleDataset()
        dataloader2 = data.DataLoader(dataset2, batch_size=2)
        model2 = SimpleNet()
        if torch.cuda.is_available():
            model2 = model2.cuda()
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

        for inputs, _ in dataloader2:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            optimizer2.zero_grad()
            outputs = model2(inputs)
            loss = outputs.mean()
            loss.backward()
            optimizer2.step()

        # Save model state
        state_dict2 = model2.state_dict()

        # Assert the model states are the same
        for key in state_dict1:
            self.assertTrue(torch.equal(state_dict1[key], state_dict2[key]))


def is_equal_instance(i1, i2):
    if isinstance(i1, torch.Tensor):
        assert isinstance(i2, torch.Tensor)
        return torch.allclose(i1, i2)
    elif isinstance(i1, np.ndarray):
        assert isinstance(i2, np.ndarray)
        return np.allclose(i1, i2)
    elif isinstance(i1, (int, float, str)):
        assert isinstance(i2, (int, float, str))
        return i1 == i2
    elif isinstance(i1, dict):
        raise NotImplementedError("Should use assert_equal_dictionaries.")
    elif isinstance(i1, (list, tuple)):
        raise NotImplementedError("Should use assert_equal_sequences.")
    elif i1 is None:
        return i2 is None
    elif isinstance(i1, bool):
        assert isinstance(i2, bool)
        return i1 == i2
    elif isinstance(i1, torch.device):
        assert isinstance(i2, torch.device)
        return i1 == i2
    else:
        raise NotImplementedError(f"Unsure how to compare type {type(i1)}.")


def is_equal_sequence(s1, s2):
    len_s1 = len(s1)
    len_s2 = len(s2)
    assert len_s1 == len_s2
    for i in range(len_s1):
        if isinstance(s1[i], dict):
            assert isinstance(s2[i], dict)
            result = is_equal_dictionary(s1[i], s2[i])
        elif isinstance(s1[i], (list, tuple)):
            result = is_equal_sequence(s1[i], s2[i])
        else:
            result = is_equal_instance(s1[i], s2[i])

        if not result:
            breakpoint()
            return False
    return True


def is_equal_dictionary(d1, d2, exceptions=()):
    symmetric_difference = set(d1.keys()) ^ set(d2.keys())
    if len(symmetric_difference) > 0:
        raise ValueError(f"Keys do not match: {symmetric_difference}")

    for k in d1.keys():
        if k in exceptions:
            continue
        if isinstance(d1[k], dict):
            assert isinstance(d2[k], dict)
            result = is_equal_dictionary(d1[k], d2[k], exceptions=exceptions)
        elif isinstance(d1[k], (list, tuple)):
            assert isinstance(d2[k], (list, tuple))
            result = is_equal_sequence(d1[k], d2[k])
        else:
            result = is_equal_instance(d1[k], d2[k])
        if not result:
            breakpoint()
            return False
    return True


class TestEndtoEndReproducibility(unittest.TestCase):
    def test_reproducibility(self):
        # Remove runs directory
        example_project_dir = "../../src/example_project/"
        runs_dir = os.path.join(example_project_dir, "runs")
        shutil.rmtree(runs_dir, ignore_errors=True)

        runs_dir = os.path.join(os.getcwd(), runs_dir, "devrun")

        # Command 1
        c1 = f"cd {example_project_dir} && PYTHONPATH=src/example_project/ devrun config/example_config.py --fit.max_steps=8"
        stdout = subprocess.getoutput(c1)
        print(stdout)
        run_dir_c1 = os.path.join(runs_dir, os.listdir(runs_dir)[0])

        # Command 2
        print(f"Example run wrote to: {run_dir_c1}")
        c2 = f"cd {example_project_dir} && PYTHONPATH=src/example_project/ devrun config/example_config.py --fit.max_steps=16 --continue={run_dir_c1}"
        stdout = subprocess.getoutput(c2)
        print(stdout)

        # Command 3
        c3 = f"cd {example_project_dir} && PYTHONPATH=src/example_project/ devrun config/example_config.py --fit.max_steps=16"
        stdout = subprocess.getoutput(c3)
        print(stdout)

        sorted_runs_dir = sorted(os.listdir(runs_dir))

        run_dir_c2 = os.path.join(runs_dir, sorted_runs_dir[1])
        run_dir_c3 = os.path.join(runs_dir, sorted_runs_dir[2])

        checkpoint_c2 = checkpoint.get_sorted_checkpoints(
            os.path.join(run_dir_c2, "files", "checkpoints")
        )[-1]
        checkpoint_c3 = checkpoint.get_sorted_checkpoints(
            os.path.join(run_dir_c3, "files", "checkpoints")
        )[-1]

        print(f"Checkpoint from run 2: {checkpoint_c2}")
        print(f"Checkpoint from run 3: {checkpoint_c3}")

        state_dict_c2 = torch.load(
            os.path.join(run_dir_c2, "files", "checkpoints", checkpoint_c2)
        )
        state_dict_c3 = torch.load(
            os.path.join(run_dir_c3, "files", "checkpoints", checkpoint_c3)
        )

        self.assertTrue(
            is_equal_dictionary(state_dict_c2["model"], state_dict_c3["model"])
        )
        self.assertTrue(
            is_equal_dictionary(state_dict_c2["optimizer"], state_dict_c3["optimizer"])
        )
        self.assertTrue(
            is_equal_dictionary(
                state_dict_c2["train_state"], state_dict_c3["train_state"]
            )
        )
        self.assertTrue(
            is_equal_dictionary(
                state_dict_c2["random_state"], state_dict_c3["random_state"]
            )
        )


if __name__ == "__main__":
    unittest.main()
