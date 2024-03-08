import os
from collections import defaultdict
from os.path import basename, commonprefix, dirname, relpath

import numpy as np
import torch
from rail1 import utils
from torch.utils import data


def get_dictionary_length(dictionary, length=None):
    for k in dictionary:
        if isinstance(dictionary[k], dict):
            length = get_dictionary_length(dictionary[k], length)

        else:
            if length is None:
                length = len(dictionary[k])
            else:
                assert length == len(
                    dictionary[k]
                ), f"Expected {length}, got {len(dictionary[k])} for {k}."

    return length


class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point


class MemmapDataset:
    def __init__(self, root):

        self.files = sorted(list(utils.rglob(root, "*.npy")))
        self.dataset = self.construct_dataset()
        self.length = get_dictionary_length(self.dataset)

    def construct_dataset(self):

        def add_to_dict(file_path, path_components, dictionary):

            if len(path_components) == 1:
                memmap = np.lib.format.open_memmap(file_path, mode="c")
                # print(len(memmap))
                dictionary[path_components[0]] = memmap
            else:
                if path_components[0] not in dictionary:
                    dictionary[path_components[0]] = {}

                add_to_dict(
                    file_path, path_components[1:], dictionary[path_components[0]]
                )

        common_path = commonprefix(self.files)

        file_dict = {}
        for file_path in self.files:
            relative_path = relpath(file_path, common_path)
            path_components = relative_path.split("/")

            path_components[-1] = basename(path_components[-1]).split(".")[0]

            add_to_dict(file_path, path_components, file_dict)

        return file_dict


class ShapeNetMemmap(data.Dataset):

    def __init__(self, root, split):
        self.root = root

        category = self.root.split("/")[-1]
        self.category = category

        self.memmap = MemmapDataset(root)

        split_filename = os.path.join(self.root, f"{split}.lst")
        index_mapping_filename = os.path.join(
            self.root, f"{category}_index_mapping.txt"
        )
        with open(index_mapping_filename, "r") as f:
            index_mapping = dict([line.strip().split("\t") for line in f.readlines()])
        index_mapping = {v: int(k) for k, v in index_mapping.items()}
        with open(split_filename, "r") as f:
            split = sorted([line.strip().split(".")[0] for line in f.readlines()])
        self.indices = sorted([index_mapping[shape_id] for shape_id in split])

        surface_index_mapping_filename = os.path.join(
            self.root, "4_pointcloud", f"{category}_index_mapping.txt"
        )
        with open(surface_index_mapping_filename, "r") as f:
            index_mapping = dict([line.strip().split("\t") for line in f.readlines()])
        index_mapping = {
            os.path.splitext(v)[0]: int(k) for k, v in index_mapping.items()
        }

        self.surface_indices = [index_mapping[shape_id] for shape_id in split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        points_idx = self.indices[idx]
        surface_idx = self.surface_indices[idx]

        near_label = self.memmap.dataset[f"{self.category}_near_label"][points_idx]
        near_points = self.memmap.dataset[f"{self.category}_near_points"][points_idx]
        scale = self.memmap.dataset[f"{self.category}_scale"][points_idx]
        vol_label = self.memmap.dataset[f"{self.category}_vol_label"][points_idx]
        vol_points = self.memmap.dataset[f"{self.category}_vol_points"][points_idx]

        surface_loc = self.memmap.dataset["4_pointcloud"][f"{self.category}_loc"][
            surface_idx
        ]
        surface_normals = self.memmap.dataset["4_pointcloud"][
            f"{self.category}_normals"
        ][surface_idx]
        surface_points = self.memmap.dataset["4_pointcloud"][f"{self.category}_points"][
            surface_idx
        ]
        surface_scale = self.memmap.dataset["4_pointcloud"][f"{self.category}_scale"][
            surface_idx
        ]

        return {
            "near_label": near_label,
            "near_points": near_points,
            "scale": scale,
            "vol_label": vol_label,
            "vol_points": vol_points,
            "surface_loc": surface_loc,
            "surface_normals": surface_normals,
            "surface_points": surface_points,
            "surface_scale": surface_scale,
        }


class ShapeNet:
    def __init__(
        self,
        split,
        categories=["02954340"],
        transform=None,
        sampling=True,
        num_samples=4096,
        return_surface=True,
        surface_sampling=True,
        pc_size=2048,
        replica=16,
    ):

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        dataset_folder = os.environ['DATAROOT']

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, "ShapeNetV2_point")
        self.mesh_folder = os.path.join(self.dataset_folder, "ShapeNetV2_watertight")

        # if categories is None:
        #     categories = os.listdir(self.point_folder)
        #     categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()

        # self.models = []
        # for c_idx, c in enumerate(categories):
        #     subpath = os.path.join(self.point_folder, c)
        #     assert os.path.isdir(subpath)

        #     split_file = os.path.join(subpath, split + ".lst")
        #     with open(split_file, "r") as f:
        #         models_c = f.read().split("\n")

        #     self.models += [
        #         {"category": c, "model": m.replace(".npz", "")} for m in models_c
        #     ]

        datasets = [
            ShapeNetMemmap(self.point_folder + "/" + c, split) for c in categories
        ]
        # Concatenate datasets
        self.dataset = data.ConcatDataset(datasets)

        self.replica = replica

    def __len__(self):
        # if self.split != "train":
        #     return len(self.models)
        # else:
        #     return len(self.models) * self.replica
        return len(self.dataset)

    def __getitem__(self, idx):

        # category = self.models[idx]["category"]
        # model = self.models[idx]["model"]

        # point_path = os.path.join(self.point_folder, category, model + ".npz")
        # try:
        #     with np.load(point_path) as data:
        #         vol_points = data["vol_points"]
        #         vol_label = data["vol_label"]
        #         near_points = data["near_points"]
        #         near_label = data["near_label"]
        # except Exception as e:
        #     print(e)
        # print(point_path)

        # with open(point_path.replace(".npz", ".npy"), "rb") as f:
        #     scale = np.load(f).item()

        data = self.dataset[idx]

        if self.return_surface:
            surface = data["surface_points"].astype(np.float32)
            surface = surface * data["surface_scale"]
            if self.surface_sampling:
                ind = np.random.default_rng().choice(
                    surface.shape[0], self.pc_size, replace=False
                )
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        vol_points = data["vol_points"]
        vol_label = data["vol_label"]
        near_points = data["near_points"]
        near_label = data["near_label"]

        if self.sampling:

            ind = np.random.default_rng().choice(
                vol_points.shape[0], self.num_samples, replace=False
            )
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(
                near_points.shape[0], self.num_samples, replace=False
            )
            near_points = near_points[ind]
            near_label = near_label[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == "train":
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)

        if self.return_surface:
            return points, labels, surface  # , category_ids[category]
        else:
            return points, labels  # , category_ids[category]


#  =============================================================================
# import os
# import glob
# import random

# import yaml

# import torch
# from torch.utils import data

# import numpy as np

# from PIL import Image

# import h5py

# category_ids = {
#     '02691156': 0,
#     '02747177': 1,
#     '02773838': 2,
#     '02801938': 3,
#     '02808440': 4,
#     '02818832': 5,
#     '02828884': 6,
#     '02843684': 7,
#     '02871439': 8,
#     '02876657': 9,
#     '02880940': 10,
#     '02924116': 11,
#     '02933112': 12,
#     '02942699': 13,
#     '02946921': 14,
#     '02954340': 15,
#     '02958343': 16,
#     '02992529': 17,
#     '03001627': 18,
#     '03046257': 19,
#     '03085013': 20,
#     '03207941': 21,
#     '03211117': 22,
#     '03261776': 23,
#     '03325088': 24,
#     '03337140': 25,
#     '03467517': 26,
#     '03513137': 27,
#     '03593526': 28,
#     '03624134': 29,
#     '03636649': 30,
#     '03642806': 31,
#     '03691459': 32,
#     '03710193': 33,
#     '03759954': 34,
#     '03761084': 35,
#     '03790512': 36,
#     '03797390': 37,
#     '03928116': 38,
#     '03938244': 39,
#     '03948459': 40,
#     '03991062': 41,
#     '04004475': 42,
#     '04074963': 43,
#     '04090263': 44,
#     '04099429': 45,
#     '04225987': 46,
#     '04256520': 47,
#     '04330267': 48,
#     '04379243': 49,
#     '04401088': 50,
#     '04460130': 51,
#     '04468005': 52,
#     '04530566': 53,
#     '04554684': 54,
# }

# class ShapeNet(data.Dataset):
#     def __init__(self, dataset_folder, split, categories=['03001627'], transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=16):

#         self.pc_size = pc_size

#         self.transform = transform
#         self.num_samples = num_samples
#         self.sampling = sampling
#         self.split = split

#         self.dataset_folder = dataset_folder
#         self.return_surface = return_surface
#         self.surface_sampling = surface_sampling

#         self.dataset_folder = dataset_folder
#         self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
#         self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')

#         if categories is None:
#             categories = os.listdir(self.point_folder)
#             categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
#         categories.sort()

#         print(categories)

#         self.models = []
#         for c_idx, c in enumerate(categories):
#             subpath = os.path.join(self.point_folder, c)
#             assert os.path.isdir(subpath)

#             split_file = os.path.join(subpath, split + '.lst')
#             with open(split_file, 'r') as f:
#                 models_c = f.read().split('\n')

#             self.models += [
#                 {'category': c, 'model': m.replace('.npz', '')}
#                 for m in models_c
#             ]

#         self.replica = replica

#     def __getitem__(self, idx):
#         idx = idx % len(self.models)

#         category = self.models[idx]['category']
#         model = self.models[idx]['model']

#         point_path = os.path.join(self.point_folder, category, model+'.npz')
#         try:
#             with np.load(point_path) as data:
#                 vol_points = data['vol_points']
#                 vol_label = data['vol_label']
#                 near_points = data['near_points']
#                 near_label = data['near_label']
#         except Exception as e:
#             print(e)
#             print(point_path)

#         with open(point_path.replace('.npz', '.npy'), 'rb') as f:
#             scale = np.load(f).item()

#         if self.return_surface:
#             pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model+'.npz')
#             with np.load(pc_path) as data:
#                 surface = data['points'].astype(np.float32)
#                 surface = surface * scale
#             if self.surface_sampling:
#                 ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
#                 surface = surface[ind]
#             surface = torch.from_numpy(surface)

#         if self.sampling:
#             ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
#             vol_points = vol_points[ind]
#             vol_label = vol_label[ind]

#             ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
#             near_points = near_points[ind]
#             near_label = near_label[ind]


#         vol_points = torch.from_numpy(vol_points)
#         vol_label = torch.from_numpy(vol_label).float()

#         if self.split == 'train':
#             near_points = torch.from_numpy(near_points)
#             near_label = torch.from_numpy(near_label).float()

#             points = torch.cat([vol_points, near_points], dim=0)
#             labels = torch.cat([vol_label, near_label], dim=0)
#         else:
#             points = vol_points
#             labels = vol_label

#         if self.transform:
#             surface, points = self.transform(surface, points)

#         if self.return_surface:
#             return points, labels, surface, category_ids[category]
#         else:
#             return points, labels, category_ids[category]

#     def __len__(self):
#         if self.split != 'train':
#             return len(self.models)
#         else:
#             return len(self.models) * self.replica
