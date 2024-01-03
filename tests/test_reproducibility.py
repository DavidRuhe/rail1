import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import unittest
from rail1.utils.seed import set_seed


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


if __name__ == "__main__":
    unittest.main()
