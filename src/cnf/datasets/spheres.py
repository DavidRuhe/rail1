import numpy as np
from torch.utils import data


def sample_sphere(d, num_points, radius):
    """
    Generate random points on a d-dimensional unit sphere.

    Args:
        d (int): The dimension of the sphere.
        num_points (int): The number of points to generate.
        radius (float): The radius of the sphere.

    Returns:
        numpy.ndarray: An array of shape (num_points, d) containing the generated points.
    """

    points = np.random.randn(num_points, d)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= radius

    return points


class SpheresDataset(data.Dataset):

    def __init__(self, num_points=1024, train=True, max_radius=1.0, min_radius=0.2):
        self.num_points = num_points
        self.train = train
        self.max_radius = max_radius
        self.min_radius = min_radius

    def __getitem__(self, index):
        return sample_sphere(
            3, self.num_points, np.random.uniform(self.min_radius, self.max_radius)
        )

    def __len__(self):
        return 100000


def get_dataloaders(batch_size=32, num_workers=0, num_prefetch=0):
    train = SpheresDataset()
    train_loader = data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=num_prefetch,
    )

    return {
        "train_loader": train_loader,
        "val_loader": None,
        "test_loader": None,
    }