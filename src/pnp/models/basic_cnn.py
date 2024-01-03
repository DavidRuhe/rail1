from torch import nn


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = conv(3, 32)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(32, 64)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = conv(64, 64)
        self.relu3 = nn.ReLU()
        self.conv4 = conv(64, 64)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 512)
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, input):
        h = self.conv1(input)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.maxpool1(h)
        h = self.dropout1(h)
        h = self.conv3(h)
        h = self.relu3(h)
        h = self.conv4(h)
        h = self.relu4(h)
        h = self.maxpool2(h)
        h = self.dropout2(h)
        h = self.flatten(h)
        h = self.fc1(h)
        h = self.relu5(h)
        h = self.dropout3(h)
        h = self.fc2(h)
        return h

