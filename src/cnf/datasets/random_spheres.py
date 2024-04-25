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

    return points.astype(np.float32)


class SpheresDataset(data.Dataset):

    def __init__(self, num_points=1024, train=True, radius_rng=(0.2, 1.0), jitter=1/32):
        self.num_points = num_points
        self.train = train
        self.min_radius, self.max_radius = radius_rng
        self.jitter = jitter

    def __getitem__(self, index):
        radius = np.random.uniform(self.min_radius - self.jitter, self.max_radius + self.jitter) + np.random.uniform(-1, 1) * self.jitter
        return sample_sphere(
            3, self.num_points, radius
        ), radius, 0

    def __len__(self):
        return 1000


def load_random_spheres_dataset(num_points=1024, batch_size=32, num_workers=0, radius_rng=(0.2, 1.0)):
    train = SpheresDataset(num_points=num_points, radius_rng=radius_rng)
    train_loader = data.DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return {
        "train_loader": train_loader,
        "val_loader": None,
        "test_loader": train_loader,
    }