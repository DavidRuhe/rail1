import torch


class ReverseDataset:
    def __init__(self, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = torch.flip(inp_data, dims=(0,))
        return inp_data, labels


def get_reverse(num_classes=10):
    train_dataset = ReverseDataset(num_classes, 16, 50_000)
    val_dataset = ReverseDataset(num_classes, 16, 10_000)
    test_dataset = ReverseDataset(num_classes, 16, 10_000)
    return train_dataset, val_dataset, test_dataset
