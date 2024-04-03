import torch
from pykeops.torch import LazyTensor


def k_means(x, c, max_iter=32, threshold=1e-5):
    N, D = x.shape
    K = len(c)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    for i in range(max_iter):
        D_ij = ((x_i - c_j) ** 2).sum(-1)
        cl = D_ij.argmin(dim=1).view(-1)

        c_old = c.clone()
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        Ncl = torch.max(Ncl, torch.ones_like(Ncl))
        c /= Ncl

        diff = (c - c_old).abs().max()

        if not torch.isfinite(diff):
            print("Numerical errors at iteration", i)
            break

        if diff < threshold:
            break

    return cl, c


import torch
from pykeops.torch import LazyTensor
import time
from pykeops.torch.cluster import cluster_centroids

def KMeans(x, c, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = c.clone()  # Simplistic initialization for the centroids
    K = c.shape[0]
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------

        d_ij = torch.cdist(x, c)  # (N, K) symbolic squared distances
        cl = d_ij.argmin(dim=1)

        c_old = c.clone()

        # Update the centroids c
        c.zero_()
        c.scatter_add_(0, cl[:, None].expand(-1, D), x)
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        Ncl = torch.max(Ncl, torch.ones_like(Ncl))
        c /= Ncl
        
        diff = (c - c_old).abs().max()

        if not torch.isfinite(diff):
            print("Numerical errors at iteration", i)
            break

        if diff < 1e-5:
            break

    return cl, c


def farthest_points(x, n):
    N, D = x.shape
    centroids = torch.zeros(n, D, dtype=x.dtype, device=x.device)
    centroid = x.mean(0)

    distance = torch.full((N,), float("inf"), dtype=x.dtype, device=x.device)
    for i in range(n):
        centroids[i] = centroid
        d_ij = torch.square(x - centroid).sum(-1)
        mask = d_ij < distance
        distance[mask] = d_ij[mask]
        centroid = x[distance.argmax()]
    return centroids


if __name__ == "__main__":
    N, D, K = 1024, 2, 512
    x = torch.randn(N, D)
    c = x[:K].clone()

    cl, c = k_means(x, c)

    import matplotlib.pyplot as plt

    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=cl.numpy(), s=1)
    plt.scatter(c[:, 0].numpy(), c[:, 1].numpy(), c="red")
    plt.savefig("kmeans.png")

    print("Done")

    import time
    t0 = time.time()
    # Test farthest points
    c = farthest_points(x, K)
    t1 = time.time()
    print("Time", t1 - t0)
    cl, c = k_means(x, c)

    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=cl.numpy(), s=1)
    plt.scatter(c[:, 0].numpy(), c[:, 1].numpy(), c="red")
    plt.savefig("kmeans_farthest.png")

    print("Done")
    print("Time", time.time() - t0)

    # Test now on cuda

    x = x.cuda()
    c = c.cuda()

    


    for i in range(100):
        t0 = time.time()
        c = farthest_points(x, K)
        t1 = time.time()
        print("Time", t1 - t0)
        cl, c = k_means(x, c)
        print("Time", time.time() - t0)