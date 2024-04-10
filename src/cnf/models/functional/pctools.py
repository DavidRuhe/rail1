import torch


def index(tensor, idx):
    """
    Indexes a tensor with a multidimensional index tensor.

    E.g., for a 3D tensor (B, N, D) and a 3D index tensor (B, M, N),
    the function returns a 3D tensor (B, M, D) with the features.

    Args:
        tensor: (B, N, D) Input tensor
        idx: (B, ..., N) Index tensor
    Returns:
        new_tensor: (B, ..., D) Indexed features
    """
    B, N, D = tensor.shape
    B, *_, N = idx.shape
    view_shape = (B,) + (1,) * (len(idx.shape) - 1)
    repeat_shape = (1,) + idx.shape[1:]
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=tensor.device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_tensor = tensor[batch_indices, idx, :]
    return new_tensor


def knn(pos1, pos2, k, return_dist=False):
    """
    Finds for each point in pos1 the K nearest neighbors in pos2.

    Args:
        pos1: (B, N, D) coordinates of the first point cloud
        pos2: (B, M, D) coordinates of the second point cloud
        K: number of neighbors
        return_dist: whether to return the distances as well
    Returns:
        idx: (B, N, K) indices of the K nearest neighbors in pos2
        dist: (B, N, K) distances of the K nearest neighbors in pos2
    """
    result = torch.cdist(pos2, pos1).topk(k, dim=-1, largest=False)
    if return_dist:
        return result
    else:
        return result.indices


def ball_query(pos1, pos2, k, radius):
    """
    Finds for each point in pos1 the K neighbors in pos2 within a radius.

    If there are less than K neighbors within the radius,
    the function duplicates the closest neighbor.

    Args:
        pos1: (B, N, D) coordinates of the first point cloud
        pos2: (B, M, D) coordinates of the second point cloud
        K: number of neighbors
        radius: radius of the ball
    Returns:
        idx: (B, M, K) indices of the neighbors in pos2
    """
    B, N, D = pos1.shape
    B, M, D = pos2.shape
    group_idx = torch.arange(N, device=pos1.device)[None, None].repeat([B, M, 1])
    dist = torch.cdist(pos2, pos1)
    group_idx[dist > radius] = N
    group_idx = group_idx.sort(dim=-1).indices[..., :k]

    # If there are less than K neighbors, duplicate the closest one
    group_first = group_idx[..., :1].expand(-1, -1, k)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def recenter_groups(grouped_pos, pos):
    """
    Normalizes the coordinates of the groups in grouped_pos with respect to pos.

    I.e., for each point in pos, grouped_pos contains the coordinates of the K
    group members. The function normalizes the coordinates of the group
    members with respect to the center of the group.

    Args:
        grouped_pos: (B, M, K, D) coordinates of the groups in grouped_pos
        pos: (B, M, D) coordinates of the points in pos
    Returns:
        normalized: (B, M, K, D) normalized coordinates of the groups
    """

    B, N, K, D = grouped_pos.shape
    B, N, D = pos.shape
    return grouped_pos - pos[:, :, None]


def group_to_idx(pos_features, idx, query_fn, normalize_fn):
    """Groups features according to the given index tensor.

    For each point in the new point cloud, the function selects a set
    of centroids according to idx. Then, query_fn is used to find the
    neighbors of the centroids in the original point cloud. The function
    normalizes the coordinates of the neighbors with respect to the centroid
    and concatenates the normalized coordinates with the features.

    Args:
        pos_features: tuple of (B, N, D) coordinates and (B, N, C) features of the original point cloud
        idx: (B, M) centroid index tensor
        query_fn: function to query neighbors, e.g., knn or ball_query
        normalize_fn: function to normalize the groups, e.g., recenter_groups

    Returns:
        new_pos: (B, M, D) coordinates of the new point cloud
        new_features: (B, M, C + D) features of the new point cloud
    """
    pos, features = pos_features
    new_pos = index(pos, idx)
    group_idx = query_fn(pos, new_pos)
    grouped_pos = index(pos, group_idx)
    grouped_pos_norm = normalize_fn(grouped_pos, new_pos)
    grouped_features = index(features, group_idx)
    grouped_features = torch.cat([grouped_pos_norm, grouped_features], dim=-1)
    return new_pos, grouped_features


if __name__ == "__main__":
    x = torch.rand(2, 10, 3)
    features = torch.rand(2, 10, 5)
    y = torch.rand(2, 5, 3)
    k = 3
    idx = torch.randint(0, 10, (2, 5))
    import functools

    knn_ = functools.partial(knn, k=k)
    ball_query_ = functools.partial(ball_query, k=k, radius=0.1)
    new_pos, new_features = group_to_idx(x, features, idx, knn_, recenter_groups)
    new_pos, new_features = group_to_idx(x, features, idx, knn_, lambda x, y: x)
    new_pos, new_features = group_to_idx(x, features, idx, ball_query_, recenter_groups)
