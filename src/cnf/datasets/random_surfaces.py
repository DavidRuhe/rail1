import numpy as np
from torch.utils import data


def sample_sphere(num_points, radius, d=3):
    """
    Generate random points on a d-dimensional unit sphere.

    Args:
        num_points (int): The number of points to generate.
        radius (float): The radius of the sphere.
        d (int, optional): The dimension of the sphere. Defaults to 3.

    Returns:
        numpy.ndarray: An array of shape (num_points, d) containing the generated points.
    """

    points = np.random.randn(num_points, d)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= radius

    assert len(points) == num_points

    return points.astype(np.float32)


def sample_torus(num_points, radius, tube_radius):
    """
    Generate random points on a torus.

    Args:
        num_points (int): The number of points to generate.
        radius (float): The radius of the torus.
        tube_radius (float): The radius of the tube.

    Returns:
        numpy.ndarray: An array of shape (num_points, 3) containing the generated points.
    """

    u = np.random.rand(num_points) * 2 * np.pi
    v = np.random.rand(num_points) * 2 * np.pi

    x = (radius + tube_radius * np.cos(v)) * np.cos(u)
    y = (radius + tube_radius * np.cos(v)) * np.sin(u)
    z = tube_radius * np.sin(v)

    points = np.stack([x, y, z], axis=1).astype(np.float32)

    assert len(points) == num_points

    return points.astype(np.float32)


def sample_cilinder(num_points, r, h):
    """
    Sample points on a cylinder surface.

    Args:
        num_points (int): The number of points to sample.
        r (float): The radius of the cylinder.
        h (float): The height of the cylinder.

    Returns:
        numpy.ndarray: An array of shape (num_points, 3) containing the sampled points.
    """
    # Areas of the cylinder parts
    area_side = 2 * np.pi * r * h
    area_top = np.pi * r**2
    area_bottom = np.pi * r**2
    total_area = area_side + area_top + area_bottom

    # Probabilities for each part
    p_side = area_side / total_area
    p_top = area_top / total_area
    p_bottom = area_bottom / total_area

    # Choose which part to sample from for each point
    parts = np.random.choice(
        ["side", "top", "bottom"], size=num_points, p=[p_side, p_top, p_bottom]
    )

    points = np.zeros((num_points, 3))

    num_top = np.sum(parts == "top")
    num_bottom = np.sum(parts == "bottom")
    num_side = num_points - num_top - num_bottom

    # Sample top points uniformly in a circle
    R = np.sqrt(np.random.rand(num_top)) * r
    theta = np.random.rand(num_top) * 2 * np.pi
    top_points = np.stack(
        [R * np.cos(theta), R * np.sin(theta), np.ones(num_top) * h], axis=1
    )

    # Sample bottom points uniformly in a circle
    R = np.sqrt(np.random.rand(num_bottom)) * r
    theta = np.random.rand(num_bottom) * 2 * np.pi
    bottom_points = np.stack(
        [R * np.cos(theta), R * np.sin(theta), np.zeros(num_bottom)], axis=1
    )

    # Sample side points uniformly on the side
    u = np.random.rand(num_side) * 2 * np.pi
    v = np.random.rand(num_side) * h

    x = r * np.cos(u)
    y = r * np.sin(u)
    z = v

    side_points = np.stack([x, y, z], axis=1)

    points[parts == "top"] = top_points
    points[parts == "bottom"] = bottom_points
    points[parts == "side"] = side_points

    # Recenter to [-h, h]
    points[:, 2] -= h / 2
    assert len(points) == num_points

    return points.astype(np.float32)


def sample_cone(num_points, radius, height):
    theta = np.pi / 2  # 90 degrees

    # Calculate areas
    L = np.sqrt(radius**2 + height**2)
    surface_area_cone = np.pi * radius * L
    base_area = np.pi * radius**2
    total_area = surface_area_cone + base_area

    probability_cone = surface_area_cone / total_area
    probability_base = base_area / total_area

    parts = np.random.choice(
        ["cone", "base"], size=num_points, p=[probability_cone, probability_base]
    )

    num_points_cone = np.sum(parts == "cone")
    num_points_base = num_points - num_points_cone

    points = np.zeros((num_points, 3))

    # Generate random points on the cone surface
    random_u = np.random.uniform(0, 1, num_points_cone)
    random_r = np.sqrt(random_u) * radius
    random_phi = np.random.uniform(0, theta, num_points_cone)
    x_cone = random_r * np.cos(2 * np.pi * random_phi / theta)
    y_cone = random_r * np.sin(2 * np.pi * random_phi / theta)
    z_cone = height * (1 - random_r / radius)

    # Generate random points on the base
    random_r_base = np.sqrt(np.random.uniform(0, 1, num_points_base)) * radius
    random_phi_base = np.random.uniform(
        0, 2 * np.pi, num_points_base
    )  # Full circle for base
    x_base = random_r_base * np.cos(random_phi_base)
    y_base = random_r_base * np.sin(random_phi_base)
    z_base = np.zeros(num_points_base)  # Z-coordinates are zero at the base

    points[parts == "cone"] = np.stack([x_cone, y_cone, z_cone], axis=1)
    points[parts == "base"] = np.stack([x_base, y_base, z_base], axis=1)

    # Recenter to [-h/2, h/2]
    points[:, 2] -= height / 2
    assert len(points) == num_points

    return points.astype(np.float32)


def sample_cuboid(num_points, width, height, depth):
    """
    Sample points on a cuboid surface.

    Args:
        num_points (int): The number of points to sample.
        width (float): The width of the cuboid.
        height (float): The height of the cuboid.
        depth (float): The depth of the cuboid.

    Returns:
        numpy.ndarray: An array of shape (num_points, 3) containing the sampled points.
    """
    # Areas of the cuboid parts
    area_front = width * height
    area_back = width * height
    area_top = width * depth
    area_bottom = width * depth
    area_left = height * depth
    area_right = height * depth
    total_area = (
        area_front + area_back + area_top + area_bottom + area_left + area_right
    )

    # Probabilities for each part
    p_front = area_front / total_area
    p_back = area_back / total_area
    p_top = area_top / total_area
    p_bottom = area_bottom / total_area
    p_left = area_left / total_area
    p_right = area_right / total_area

    # Choose which part to sample from for each point
    parts = np.random.choice(
        ["front", "back", "top", "bottom", "left", "right"],
        size=num_points,
        p=[p_front, p_back, p_top, p_bottom, p_left, p_right],
    )

    points = np.zeros((num_points, 3))

    num_front = np.sum(parts == "front")
    num_back = np.sum(parts == "back")
    num_top = np.sum(parts == "top")
    num_bottom = np.sum(parts == "bottom")
    num_left = np.sum(parts == "left")
    num_right = num_points - num_front - num_back - num_top - num_bottom - num_left

    # Sample front points uniformly in a rectangle
    x = np.random.rand(num_front) * width - width / 2
    y = np.random.rand(num_front) * height - height / 2
    z = np.ones(num_front) * depth / 2
    front_points = np.stack([x, y, z], axis=1)

    # Sample back points uniformly in a rectangle
    x = np.random.rand(num_back) * width - width / 2
    y = np.random.rand(num_back) * height - height / 2
    z = np.ones(num_back) * -depth / 2
    back_points = np.stack([x, y, z], axis=1)

    # Sample top points uniformly in a rectangle
    x = np.random.rand(num_top) * width - width / 2
    y = np.ones(num_top) * height / 2
    z = np.random.rand(num_top) * depth - depth / 2
    top_points = np.stack([x, y, z], axis=1)

    # Sample bottom points uniformly in a rectangle
    x = np.random.rand(num_bottom) * width - width / 2
    y = np.ones(num_bottom) * -height / 2
    z = np.random.rand(num_bottom) * depth - depth / 2
    bottom_points = np.stack([x, y, z], axis=1)

    # Sample left points uniformly in a rectangle
    x = np.ones(num_left) * -width / 2
    y = np.random.rand(num_left) * height - height / 2
    z = np.random.rand(num_left) * depth - depth / 2
    left_points = np.stack([x, y, z], axis=1)

    # Sample right points uniformly in a rectangle
    x = np.ones(num_right) * width / 2
    y = np.random.rand(num_right) * height - height / 2
    z = np.random.rand(num_right) * depth - depth / 2
    right_points = np.stack([x, y, z], axis=1)

    points[parts == "front"] = front_points
    points[parts == "back"] = back_points
    points[parts == "top"] = top_points
    points[parts == "bottom"] = bottom_points
    points[parts == "left"] = left_points
    points[parts == "right"] = right_points
    assert len(points) == num_points

    assert np.all(points.max(axis=0) < 1)
    assert np.all(points.min(axis=0) > -1)

    return points.astype(np.float32)


class ObjectsDataset(data.Dataset):

    def __init__(
        self, num_points=1024, shapes=["sphere", "torus", "cylinder", "cone", "cuboid"]
    ):
        self.num_points = num_points
        self.shapes = shapes

    def __getitem__(self, index):

        jitter = 0.0
        jitter = np.random.rand(self.num_points, 3) * jitter - jitter / 2

        object = np.random.choice(self.shapes)
        if object == "sphere":
            radius = np.random.uniform(0.2, 0.8)
            label = 0
            points = sample_sphere(self.num_points, radius)
            points += jitter
            return points, radius, label
        elif object == "torus":
            radius = np.random.uniform(0.4, 0.8)
            tube_radius = 0.2
            label = 1
            points = sample_torus(self.num_points, radius, tube_radius)
            points += jitter
            return points, radius, label

        elif object == "cylinder":
            r = np.random.uniform(0.2, 0.8)
            # h = np.random.uniform(0.2, 1.0)
            h = 0.5
            label = 2
            points = sample_cilinder(self.num_points, r, h)
            points += jitter
            return points, r, label

        elif object == "cone":
            radius = np.random.uniform(0.2, 0.8)
            # height = np.random.uniform(0.2, 1.0)
            h = 0.5
            label = 3
            points = sample_cone(self.num_points, radius, h)
            points += jitter
            return points, radius, label

        elif object == "cuboid":
            width = np.random.uniform(0.2, 0.8)
            h = 0.5
            d = 0.5
            label = 4
            # height = np.random.uniform(0.2, 0.8)
            # depth = np.random.uniform(0.2, 0.8)
            points = sample_cuboid(self.num_points, width, h, d)
            points += jitter
            return points, width, label

        else:
            raise ValueError(f"Unknown object type: {object}")

    def __len__(self):
        return 5000


def load_random_shapes_dataset(batch_size=32, num_workers=0, *args, **kwargs):
    train = ObjectsDataset(*args, **kwargs)
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
