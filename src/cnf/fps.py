import functools

import numpy as np
import rail1
import torch
import torch.nn.functional as F
import plotly.graph_objects as go


import datasets
import models


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    farthest = torch.zeros(B, dtype=torch.long, device=device)

    centroid_locs = torch.zeros(B, npoint, 3, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        centroid_locs[:, i : i + 1] = centroid
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance  # Smaller such that we select a new point
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.to(torch.int32), centroid_locs


def forward_and_loss_fn(points, model, num_points=3):

    points = torch.cat(
        [
            points.mean(1, keepdim=True),
            points,
        ],
        dim=1,
    )
    _, targets = farthest_point_sample(points, num_points)
    preds = model.forward(points)

    loss = F.mse_loss(preds, targets, reduction="none").mean((1, 2))

    return loss.mean(0), {
        "loss": loss,
    }


@torch.no_grad()
def eval_batch(points, batch_idx, outputs, *, model, validation=False):
    if batch_idx > 0:
        return

    return outputs


def main(config):
    run_dir = config["run_dir"]
    dataset_config = config["dataset"]
    data = getattr(datasets, dataset_config.pop("name"))(**dataset_config)
    model = getattr(models, config["model"].pop("name"))(**config["model"])
    optimizer = getattr(rail1.optimizers, config["optimizer"].pop("name"))(
        model, **config["optimizer"]
    )

    device = config["device"]

    model = model.to(device)

    logging_fn = rail1.utils.get_logging_fn(
        logger=rail1.loggers.WANDBLogger() if config["wandb"] else None
    )
    metric_fns = [
        functools.partial(rail1.metrics.mean_key, key="loss"),
    ]

    rail1.fit(
        run_dir,
        model,
        optimizer,
        data,
        forward_and_loss_fn=forward_and_loss_fn,
        logging_fn=logging_fn,
        eval_batch_fn=functools.partial(
            eval_batch,
            model=model,
        ),
        metrics_fns=metric_fns,
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
