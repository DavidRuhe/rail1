
from torch import nn
from tqdm import tqdm
import os
import torch
from torchvision import datasets, transforms, models
import numpy as np
from torch.utils import data


DATAROOT = os.environ["DATAROOT"]


def get_anomaly():
    train_dataset = SetAnomalyDataset(set_size=10, split="train")
    val_dataset = SetAnomalyDataset(set_size=10, split="val")
    test_dataset = SetAnomalyDataset(set_size=10, split="test")
    return train_dataset, val_dataset, test_dataset


class SetAnomalyDataset(data.Dataset):
    def __init__(self, set_size=10, split="train"):
        """
        Inputs:
            img_feats - Tensor of shape [num_imgs, img_dim]. Represents the high-level features.
            labels - Tensor of shape [num_imgs], containing the class labels for the images
            set_size - Number of elements in a set. N-1 are sampled from one class, and one from another one.
            train - If True, a new set will be sampled every time __getitem__ is called.
        """
        super().__init__()
        self.set_size = set_size - 1  # The set size is here the size of correct images
        self.split = split

        self.img_feats, self.labels = self.get_features()


        self.set_size = set_size - 1  # The set size is here the size of correct images
        self.train = split == "train"
        self.train_image_set, self.test_image_set = get_datasets()

        # Tensors with indices of the images per class
        self.num_labels = self.labels.max() + 1
        self.img_idx_by_label = torch.argsort(self.labels).reshape(self.num_labels, -1)

        if not self.train:
            self.test_sets = self._create_test_sets()

    def _create_test_sets(self):
        # Pre-generates the sets for each image for the test set
        test_sets = []
        num_imgs = self.img_feats.shape[0]
        np.random.seed(42)
        test_sets = [self.sample_img_set(self.labels[idx]) for idx in range(num_imgs)]
        test_sets = torch.stack(test_sets, dim=0)
        return test_sets

    def sample_img_set(self, anomaly_label):
        """
        Samples a new set of images, given the label of the anomaly.
        The sampled images come from a different class than anomaly_label
        """
        # Sample class from 0,...,num_classes-1 while skipping anomaly_label as class
        set_label = np.random.randint(self.num_labels - 1)
        if set_label >= anomaly_label:
            set_label += 1

        # Sample images from the class determined above
        img_indices = np.random.choice(
            self.img_idx_by_label.shape[1], size=self.set_size, replace=False
        )
        img_indices = self.img_idx_by_label[set_label, img_indices]
        return img_indices

    def get_features(self):
        base_dir = os.path.join(DATAROOT, "anomaly")
        train_dataset, test_dataset = get_datasets()
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)

        train_feats_file = os.path.join(DATAROOT, "anomaly", "train_set_features.tar")
        train_set_feats = torch.load(train_feats_file)
        train_set = train_dataset

        test_feats_file = os.path.join(DATAROOT, "anomaly", "test_set_features.tar")
        test_set_feats = torch.load(test_feats_file)
        test_set = test_dataset

        labels = train_set.targets

        # Get indices of images per class
        labels = torch.LongTensor(labels)
        num_labels = labels.max() + 1
        sorted_indices = torch.argsort(labels).reshape(
            num_labels, -1
        )  # [classes, num_imgs per class]

        num_labels = labels.max() + 1

        # Determine number of validation images per class
        num_val_exmps = sorted_indices.shape[1] // 10

        # Get image indices for validation and training
        val_indices = sorted_indices[:, :num_val_exmps].reshape(-1)
        train_indices = sorted_indices[:, num_val_exmps:].reshape(-1)

        # Group corresponding image features and labels
        train_feats, train_labels = (
            train_set_feats[train_indices],
            labels[train_indices],
        )
        val_feats, val_labels = train_set_feats[val_indices], labels[val_indices]

        if self.split == "train":
            return train_feats, train_labels
        elif self.split == "val":
            return val_feats, val_labels
        else:
            return test_set_feats, torch.LongTensor(test_set.targets)

    def __len__(self):
        return self.img_feats.shape[0]

    def __getitem__(self, idx):
        anomaly = self.img_feats[idx]
        if self.train:  # If train => sample
            img_indices = self.sample_img_set(self.labels[idx])
        else:  # If test => use pre-generated ones
            img_indices = self.test_sets[idx]

        # Concatenate images. The anomaly is always the last image for simplicity
        img_set = torch.cat([self.img_feats[img_indices], anomaly[None]], dim=0)
        indices = torch.cat([img_indices, torch.LongTensor([idx])], dim=0)
        label = img_set.shape[0] - 1

        label = torch.tensor(label)

        # We return the indices of the images for visualization purpose. "Label" is the index of the anomaly
        return img_set, indices, label


def get_datasets():
    DATA_MEANS = np.array([0.485, 0.456, 0.406])
    DATA_STD = np.array([0.229, 0.224, 0.225])

    # # As torch tensors for later preprocessing
    # TORCH_DATA_MEANS = torch.from_numpy(DATA_MEANS).view(1, 3, 1, 1)
    # TORCH_DATA_STD = torch.from_numpy(DATA_STD).view(1, 3, 1, 1)

    # Resize to 224x224, and normalize to ImageNet statistic
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEANS, DATA_STD),
        ]
    )
    # Loading the training dataset.
    train_set = datasets.CIFAR100(
        root=DATAROOT, train=True, transform=transform, download=True
    )

    # Loading the test set
    test_set = datasets.CIFAR100(
        root=DATAROOT, train=False, transform=transform, download=True
    )
    return train_set, test_set


@torch.no_grad()
def extract_features(dataset, save_file, device=torch.device("cuda")):
    pretrained_model = models.resnet34(weights="IMAGENET1K_V1")
    pretrained_model.fc = nn.Sequential()
    pretrained_model.classifier = nn.Sequential()
    # To GPU
    pretrained_model = pretrained_model.to(device)

    # Only eval, no gradient required
    pretrained_model.eval()
    for p in pretrained_model.parameters():
        p.requires_grad = False

    data_loader = data.DataLoader(
        dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4
    )
    extracted_features = []
    for imgs, _ in tqdm(data_loader):
        imgs = imgs.to(device)
        feats = pretrained_model(imgs)
        extracted_features.append(feats)
    extracted_features = torch.cat(extracted_features, dim=0)
    extracted_features = extracted_features.detach().cpu()
    return extracted_features


if __name__ == "__main__":
    train_dataset, test_dataset = get_datasets()
    train_feats_file = os.path.join(DATAROOT, "anomaly", "train_set_features.tar")
    test_feats_file = os.path.join(DATAROOT, "anomaly", "test_set_features.tar")

    os.makedirs(os.path.dirname(train_feats_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_feats_file), exist_ok=True)

    train_features = extract_features(train_dataset, train_feats_file)
    test_features = extract_features(test_dataset, test_feats_file)

    torch.save(train_features, train_feats_file)
    torch.save(test_features, test_feats_file)
