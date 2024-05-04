import numpy as np
from torch.utils import data
import trimesh
import os


class Bunny:

    def __init__(self, num_points=1024):
        self.num_points = num_points
        self.path = os.path.join(os.environ["DATAROOT"], "bunny.obj")

        mesh = trimesh.load(self.path)

        centroid = mesh.centroid
        mesh.apply_translation(-centroid)
        scale_factor = max(mesh.extents)
        mesh.apply_scale(1 / scale_factor)

        self.mesh = mesh

    def __getitem__(self, index):
        points, face_index = trimesh.sample.sample_surface(self.mesh, self.num_points)

        assert np.linalg.norm(points, axis=1).max() <= 1
        normals = self.mesh.face_normals[face_index]
        negative_points = np.random.uniform(-1, 1, size=(self.num_points, 3))
        return points, normals, negative_points
    
    def __len__(self):
        return 1024


def load_bunny_dataset():
    train = Bunny()
    train_loader = data.DataLoader(
        train,
        batch_size=32,
    )
    return {
        "train_loader": train_loader,
        "val_loader": None,
        "test_loader": train_loader,
    }
