import os
import random

import numpy
import numpy as np
import torch
from rail1.callbacks.checkpoint import checkpoint, load_checkpoint
from rail1.utils.seed import set_seed
from torch import nn, optim
from torch.utils import data


# # Simple feed-forward neural network
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(10, 50)  # Assuming input features are of size 10
#         self.fc2 = nn.Linear(
#             50, 1
#         )  # Output layer (e.g., for regression or binary classification)

#     def reset_parameters(self):
#         for m in self.children():
#             if hasattr(m, "reset_parameters"):
#                 m.reset_parameters()  # type: ignore

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # Flatten the 28x28 images
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=32)


def train_model(model, optimizer, steps):
    model.train()
    for _ in range(steps):
        inputs = torch.randn(32, 10).cuda()  # Batch size of 32
        targets = torch.randn(32, 1).cuda()  # Corresponding targets

        # Forward pass
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(
            outputs, targets
        )  # Mean Squared Error Loss for simplicity

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _print_models_equal(model1, model2):
    for p in model1.state_dict():
        v1 = model1.state_dict()[p]
        v2 = model2.state_dict()[p]

        print(torch.equal(v1, v2))


set_seed(0, deterministic=True)

# Initialize models and optimizer
model1 = SimpleModel().cuda()
optimizer1 = optim.Adam(model1.parameters())

# Train model1 for a few steps and checkpoint
train_model(model1, optimizer1, steps=64)


set_seed(0, deterministic=True)
model2 = SimpleModel().cuda()
optimizer2 = optim.Adam(model2.parameters())


# Train model1 for a few steps and checkpoint
train_model(model2, optimizer2, steps=32)
random_state_dict = {
    "torch": torch.get_rng_state(),
    "numpy": numpy.random.get_state(),
    "random": random.getstate(),
    "cuda": torch.cuda.get_rng_state(),
    "cuda_all": torch.cuda.get_rng_state_all(),
}

model2_state_dict = model2.state_dict()
optimizer2_state_dict = optimizer2.state_dict()

checkpoint = {
    "model": model2_state_dict,
    "optimizer": optimizer2_state_dict,
    "random_state": random_state_dict,
}
torch.save(checkpoint, "./checkpoint.pt")
del checkpoint


set_seed(0)
model2.reset_parameters()

checkpoint = torch.load("./checkpoint.pt")
model2_state_dict = checkpoint["model"]
model2.load_state_dict(model2_state_dict)
optimizer2_state_dict = checkpoint["optimizer"]
optimizer2.load_state_dict(optimizer2_state_dict)
random_state_dict = checkpoint["random_state"]

torch.set_rng_state(random_state_dict["torch"])
torch.cuda.set_rng_state(random_state_dict["cuda"])
torch.cuda.set_rng_state_all(random_state_dict["cuda_all"])
numpy.random.set_state(random_state_dict["numpy"])
random.setstate(random_state_dict["random"])

train_model(model2, optimizer2, steps=32)

_print_models_equal(model1, model2)
