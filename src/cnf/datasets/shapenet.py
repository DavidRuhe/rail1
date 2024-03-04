import numpy as np
import os
from torchvision import transforms
from torch.utils import data
import yaml


class Field(object):
    """Data fields class."""

    def load(self, data_path, idx, category):
        """Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        """
        raise NotImplementedError

    def check_complete(self, files):
        """Checks if set is complete.

        Args:
            files: files
        """
        raise NotImplementedError


class PointsField(Field):
    """Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    """

    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files

    def load(self, model_path, idx, category, mode):
        """Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        dataset_name = model_path.split("/")[1]

        if "new" in dataset_name:
            model_path = model_path.replace(dataset_name, "ShapeNet")
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(
                model_path, self.file_name, "%s_%02d.npz" % (self.file_name, num)
            )

        try:
            points_dict = np.load(file_path)
        except:
            import ipdb

            ipdb.set_trace()
        points = points_dict["points"]
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        if "df_value" in points_dict:
            distance_value = points_dict["df_value"]
            distance_value = distance_value.astype(np.float32)
        else:  # ShapeNet
            distance_path = os.path.join(model_path, "df.npy")
            distance_value = np.load(distance_path).astype(np.float32)

        data = {
            None: points,
            "df": distance_value,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


class SubsamplePoints(object):
    """Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        points = data[None]

        if ("occ" in data) and ("df" not in data):
            occ = data["occ"]

            data_out = data.copy()
            if isinstance(self.N, int):
                idx = np.random.randint(points.shape[0], size=self.N)
                data_out.update(
                    {
                        None: points[idx, :],
                        "occ": occ[idx],
                    }
                )
            else:
                Nt_out, Nt_in = self.N
                occ_binary = occ >= 0.5
                points0 = points[~occ_binary]
                points1 = points[occ_binary]

                idx0 = np.random.randint(points0.shape[0], size=Nt_out)
                idx1 = np.random.randint(points1.shape[0], size=Nt_in)

                points0 = points0[idx0, :]
                points1 = points1[idx1, :]
                points = np.concatenate([points0, points1], axis=0)

                occ0 = np.zeros(Nt_out, dtype=np.float32)
                occ1 = np.ones(Nt_in, dtype=np.float32)
                occ = np.concatenate([occ0, occ1], axis=0)

                volume = occ_binary.sum() / len(occ_binary)
                volume = volume.astype(np.float32)

                data_out.update(
                    {
                        None: points,
                        "occ": occ,
                        "volume": volume,
                    }
                )

        elif ("df" in data) and ("occ" not in data):
            df = data["df"]

            data_out = data.copy()
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update(
                {
                    None: points[idx, :],
                    "df": df[idx],
                }
            )
        elif ("df" in data) and ("occ" in data):
            occ = data["occ"]
            df = data["df"]

            data_out = data.copy()
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update(
                {
                    None: points[idx, :],
                    "occ": occ[idx],
                    "df": df[idx],
                }
            )

        return data_out


import numpy as np


# Transforms
class PointcloudNoise(object):
    """Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    """

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dictionary): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    """Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        """Calls the transformation.

        Args:
            data (dict): data dictionary
        """
        data_out = data.copy()
        points = data[None]
        # normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]
        if "normals" in data:
            normals = data["normals"]
            data_out["normals"] = normals[indices, :]

        return data_out


class PartialPointCloudField(Field):
    """Partial Point cloud field.

    It provides the field used for partial point cloud data. These are the points
    randomly sampled on the mesh and a bounding box with random size is applied.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
        part_ratio (float): max ratio for the remaining part
    """

    def __init__(
        self,
        file_name,
        transform=None,
        multi_files=None,
        part_ratio=0.5,
        partial_type="centery_random",
    ):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.part_ratio = part_ratio
        self.partial_type = partial_type

    def load(self, model_path, idx, category, mode):
        """Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        if mode in ["val", "test"]:  # fix the size in evaluation
            self.partial_type = (
                "centerz" if "centerz" in self.partial_type else "centery"
            )
            self.part_ratio = 0.5

        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(
                model_path, self.file_name, "%s_%02d.npz" % (self.file_name, num)
            )

        try:
            pointcloud_dict = np.load(file_path)
        except:
            print("Wrong file:", file_path)

        try:
            points = pointcloud_dict["points"].astype(np.float32)
        except:
            points = pointcloud_dict["arr_0"].astype(np.float32)

        if "centery" in self.partial_type:

            if self.partial_type == "centery_random":
                random_ratio = self.part_ratio * np.random.random()
            else:
                random_ratio = self.part_ratio

            # y is up-axis
            min_x = points[:, 0].min()
            max_x = points[:, 0].max()

            min_z = points[:, 2].min()
            max_z = points[:, 2].max()

            remove_size_x = (max_x - min_x) * random_ratio
            remove_size_z = (max_z - min_z) * random_ratio

            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2
            start_x = center_x - (remove_size_x / 2)
            start_z = center_z - (remove_size_z / 2)

            crop_x_idx = np.where(
                (points[:, 0] < (start_x + remove_size_x)) & (points[:, 0] > start_x)
            )[0]
            crop_z_idx = np.where(
                (points[:, 2] < (start_z + remove_size_z)) & (points[:, 2] > start_z)
            )[0]

            crop_idx = np.intersect1d(crop_x_idx, crop_z_idx)

            valid_mask = np.ones(len(points), dtype=bool)
            valid_mask[crop_idx] = 0

            remain_points = points[valid_mask]

            data = {
                None: remain_points,
            }

        elif "centerz" in self.partial_type:

            if self.partial_type == "centerz_random":
                random_ratio = self.part_ratio * np.random.random()
            else:
                random_ratio = self.part_ratio

            # z is up-axis
            min_x = points[:, 0].min()
            max_x = points[:, 0].max()

            min_y = points[:, 1].min()
            max_y = points[:, 1].max()

            random_ratio = self.part_ratio * np.random.random()

            remove_size_x = (max_x - min_x) * random_ratio
            remove_size_y = (max_y - min_y) * random_ratio

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            start_x = center_x - (remove_size_x / 2)
            start_y = center_y - (remove_size_y / 2)

            crop_x_idx = np.where(
                (points[:, 0] < (start_x + remove_size_x)) & (points[:, 0] > start_x)
            )[0]
            crop_y_idx = np.where(
                (points[:, 1] < (start_y + remove_size_y)) & (points[:, 1] > start_y)
            )[0]

            crop_idx = np.intersect1d(crop_x_idx, crop_y_idx)

            valid_mask = np.ones(len(points), dtype=bool)
            valid_mask[crop_idx] = 0

            remain_points = points[valid_mask]

            data = {
                None: remain_points,
            }

        elif self.partial_type == "randomy_random":
            # random location, random size
            min_x = points[:, 0].min()
            max_x = points[:, 0].max()

            min_z = points[:, 2].min()
            max_z = points[:, 2].max()

            random_ratio = self.part_ratio * np.random.random()

            remove_size_x = (max_x - min_x) * random_ratio
            remove_size_z = (max_z - min_z) * random_ratio

            start_x = min_x + (max_x - min_x - remove_size_x) * np.random.random()
            start_z = min_z + (max_z - min_z - remove_size_z) * np.random.random()

            crop_x_idx = np.where(
                (points[:, 0] < (start_x + remove_size_x)) & (points[:, 0] > start_x)
            )[0]
            crop_z_idx = np.where(
                (points[:, 2] < (start_z + remove_size_z)) & (points[:, 2] > start_z)
            )[0]

            crop_idx = np.intersect1d(crop_x_idx, crop_z_idx)

            valid_mask = np.ones(len(points), dtype=bool)
            valid_mask[crop_idx] = 0

            remain_points = points[valid_mask]

            data = {
                None: remain_points,
            }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        """Check if field is complete.

        Args:
            files: files
        """
        complete = self.file_name in files
        return complete


def get_inputs_field(
    mode,
    input_type,
    pointcloud_n,
    pointcloud_noise,
    pointcloud_file,
    multi_files,
    part_ratio,
    partial_type,
    **kwargs,
):
    """Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    """
    if input_type is None:
        inputs_field = None
    elif input_type == "pointcloud":
        raise NotImplementedError
        transform = transforms.Compose(
            [
                data.SubsamplePointcloud(cfg["data"]["pointcloud_n"]),
                data.PointcloudNoise(cfg["data"]["pointcloud_noise"]),
            ]
        )
        inputs_field = data.PointCloudField(
            cfg["data"]["pointcloud_file"],
            transform,
            multi_files=cfg["data"]["multi_files"],
        )
    elif input_type == "partial_pointcloud":
        transform = transforms.Compose(
            [
                SubsamplePointcloud(pointcloud_n),
                PointcloudNoise(pointcloud_noise),
            ]
        )
        inputs_field = PartialPointCloudField(
            pointcloud_file,
            transform,
            multi_files=multi_files,
            part_ratio=part_ratio,
            partial_type=partial_type,
        )
    elif input_type == "idx":
        raise NotImplementedError
        inputs_field = data.IndexField()
    else:
        raise ValueError("Invalid input type (%s)" % input_type)
    return inputs_field


def get_data_fields(
    mode,
    points_subsample,
    input_type,
    points_file,
    points_iou_file,
    multi_files,
    **kwargs,
):
    """Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """
    points_transform = SubsamplePoints(points_subsample)

    fields = {}
    if points_file is not None:
        fields["points"] = PointsField(
            points_file, points_transform, unpackbits=False, multi_files=multi_files
        )

    if mode in ("val", "test"):
        if points_iou_file is not None:
            fields["points_iou"] = PointsField(
                points_iou_file, unpackbits=False, multi_files=multi_files
            )

    return fields


def rotate_pointcloud(pointcloud, points, points_iou=None):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)

    if points_iou is not None:
        points_iou[:, [0, 2]] = points_iou[:, [0, 2]].dot(rotation_matrix)

    return pointcloud, points, points_iou


def translate_pointcloud(pointcloud, points, points_iou=None):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype("float32")

    points = np.add(np.multiply(points, xyz1), xyz2).astype("float32")

    if points_iou is not None:
        points_iou = np.add(np.multiply(points_iou, xyz1), xyz2).astype("float32")

    return pointcloud, points, points_iou


def single_translate_pointcloud(
    pointcloud, points, points_iou=None, points_df=None, points_iou_df=None
):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[1])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )

    translated_points = np.add(np.multiply(points, xyz1), xyz2).astype("float32")

    translated_points_df = None
    if points_df is not None:
        translated_points_df = np.multiply(points_df, xyz1)

    translated_points_iou_df = None
    if points_iou_df is not None:
        translated_points_iou_df = np.multiply(points_iou_df, xyz1)

    if points_iou is not None:
        translated_points_iou = np.add(np.multiply(points_iou, xyz1), xyz2).astype(
            "float32"
        )
        return (
            translated_pointcloud,
            translated_points,
            translated_points_iou,
            translated_points_df,
            translated_points_iou_df,
        )
    else:
        return translated_pointcloud, translated_points, translated_points_df


class Shapes3dDataset(data.Dataset):
    """3D Shapes dataset class."""

    def __init__(
        self,
        dataset_folder,
        fields,
        split=None,
        categories=None,
        transform=None,
    ):
        """Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        """
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.transform = transform
        self.split = split
        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [
                c for c in categories if os.path.isdir(os.path.join(dataset_folder, c))
            ]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, "metadata.yaml")

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.metadata = {c: {"id": c, "name": "n/a"} for c in categories}

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]["idx"] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                print("Category %s does not exist in dataset." % c)

            if split is None:
                self.models += [
                    {"category": c, "model": m}
                    for m in [
                        d
                        for d in os.listdir(subpath)
                        if (os.path.isdir(os.path.join(subpath, d)) and d != "")
                    ]
                ]

            else:
                split_file = os.path.join(subpath, split + ".lst")
                with open(split_file, "r") as f:
                    models_c = f.read().split("\n")

                if "" in models_c:
                    models_c.remove("")

                self.models += [{"category": c, "model": m} for m in models_c]

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.models)

    def __getitem__(self, idx):
        """Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        """
        category = self.models[idx]["category"]
        model = self.models[idx]["model"]
        c_idx = self.metadata[category]["idx"]
        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        info = c_idx

        for field_name, field in self.fields.items():
            field_data = field.load(model_path, idx, info, self.split)

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data["%s.%s" % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transforms(data, transform_type=self.transform)

        return data

    def transforms(self, data, transform_type=None):

        if "rotate" in transform_type:
            data["inputs"], data["points"], data["points_iou"] = rotate_pointcloud(
                pointcloud=data["inputs"],
                points=data["points"],
                points_iou=data.get("points_iou"),
            )

        if "translate" in transform_type:
            data["inputs"], data["points"], data["points_iou"] = translate_pointcloud(
                pointcloud=data["inputs"],
                points=data["points"],
                points_iou=data.get("points_iou"),
            )

        if "single_trans" in transform_type:
            raise NotImplementedError
            points_df = None
            if "points.df" in data.keys():
                points_df = data["points.df"]
            points_iou_df = None
            if "points_iou.df" in data.keys():
                points_iou_df = data["points_iou.df"]

            if data.get("points_iou") is not None:
                (
                    data["inputs"],
                    data["points"],
                    data["points_iou"],
                    points_df,
                    points_iou_df,
                ) = single_translate_pointcloud(
                    pointcloud=data["inputs"],
                    points=data["points"],
                    points_iou=data["points_iou"],
                    points_df=points_df,
                    points_iou_df=points_iou_df,
                )
                if points_df is not None:
                    data["points.df"] = points_df
                if points_iou_df is not None:
                    data["points_iou.df"] = points_iou_df
            else:
                data["inputs"], data["points"], points_df = single_translate_pointcloud(
                    pointcloud=data["inputs"],
                    points=data["points"],
                    points_df=points_df,
                )
                if points_df is not None:
                    data["points.df"] = points_df

        # clean None type keys
        filtered = {k: v for k, v in data.items() if v is not None}
        data.clear()
        data.update(filtered)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        """Tests if model is complete.

        Args:
            model (str): modelname
        """
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                print('Field "%s" is incomplete: %s' % (field_name, model_path))
                return False

        return True


class IndexField(Field):
    """Basic index field."""

    def load(self, model_path, idx, category, mode):
        """Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        """
        return idx

    def check_complete(self, files):
        """Check if field is complete.

        Args:
            files: files
        """
        return True


# Datasets
def load_shapenet(
    split,
    mode,
    return_idx,
    augment_rotate,
    augment_translate,
    augment_single_trans,
    dataset_folder,
    categories=None,
    **kwargs,
):
    method = kwargs["method"]
    assert method == "shapenet_dfnet", f"Method {method} not implemented"
    fields = get_data_fields(mode, **kwargs)
    # Input fields
    inputs_field = get_inputs_field(mode, **kwargs)

    if inputs_field is not None:
        fields["inputs"] = inputs_field

    if return_idx:
        fields["idx"] = IndexField()

    if mode == "train":
        transform = []
        if augment_rotate:
            transform.append("rotate")
        if augment_translate:
            transform.append("translate")
        if augment_single_trans:
            transform.append("single_trans")
    else:
        transform = []

    dataset = Shapes3dDataset(
        dataset_folder,
        fields,
        split=split,
        categories=categories,
        transform=transform,
    )

    return dataset
