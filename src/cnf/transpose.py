import functools

import numpy as np
import rail1
import torch
import torch.nn.functional as F
import plotly.graph_objects as go


import datasets
import models


def forward_and_loss_fn(points, model, num_points=3):

    B, M, D = points.shape
    points = points[:, :2, None]
    targets = points.transpose(1, 2)

    preds = model.forward(points)

    loss = F.mse_loss(preds, targets, reduction="none")

    loss = loss.mean((1, 2, 3))

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
