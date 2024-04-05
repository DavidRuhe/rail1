import numpy as np

from .pctools import _fps_sampling, _k_means


def fps_sampling(
    pc: np.ndarray, n_samples: int, start_coord: np.ndarray = None
) -> np.ndarray:
    """
    Vanilla FPS sampling.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        n_samples (int): Number of samples.
        start_coord (np.ndarray, optional): Starting coordinate of shape (D,).
            If None, a random point from pc is chosen as the starting coordinate.

    Returns:
        np.ndarray: The selected coordinates of shape (n_samples, D).
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pc.ndim == 2, "pc should be a 2D array"
    n_pts, D = pc.shape
    assert n_pts > n_samples, "n_pts should be greater than n_samples"

    pc = pc.astype(np.float32)
    # Best performance with Fortran array
    pc = np.asfortranarray(pc)

    if start_coord is None:
        start_idx = np.random.randint(low=0, high=n_pts)
        start_coord = pc[start_idx]
    else:
        assert start_coord.shape == (D,), "start_coord should have the shape (D,)"

    # Call the Rust-implemented function
    selected_coords = _fps_sampling(pc, n_samples, start_coord)
    return selected_coords


def k_means(
    pc: np.ndarray, centroids: np.ndarray, iters: int, tolerance: float
) -> np.ndarray:
    """
    K-means clustering with a stopping criterion based on centroid shift.

    Args:
        pc (np.ndarray): The input point cloud of shape (n_pts, D).
        centroids (np.ndarray): The initial centroids of shape (n_centroids, D).
        iters (int): Maximum number of iterations.
        tolerance (float): The tolerance for centroid shift to stop the iterations early.

    Returns:
        np.ndarray: The cluster indices of shape (n_pts,).
    """
    assert pc.ndim == 2, "Input point cloud must be 2D (n_pts, D)."
    assert centroids.ndim == 2, "Centroids must be 2D (n_centroids, D)."
    assert (
        pc.shape[1] == centroids.shape[1]
    ), "Point cloud dimensions must match centroid dimensions."

    pc = pc.astype(np.float32)
    centroids = centroids.astype(np.float32)
    pc = np.asfortranarray(pc)
    centroids = np.asfortranarray(centroids)

    # Pass the tolerance parameter to the _k_means function
    return _k_means(pc, centroids, iters, tolerance)
