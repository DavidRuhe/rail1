from torch import nn


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)



class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=4, padding=1)  # 8
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=4, padding=1) # 2
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 1
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()  # 8
        self.fc1 = nn.Linear(8, 8) 
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8, 10)

    def forward(self, input):
        h = self.conv1(input)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.maxpool(h)
        h = self.dropout(h)
        h = self.flatten(h)
        h = self.fc1(h)
        h = self.relu3(h)
        h = self.dropout2(h)
        h = self.fc2(h)
        return h


