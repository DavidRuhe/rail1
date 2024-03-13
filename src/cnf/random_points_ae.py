import functools

import matplotlib.pyplot as plt
import numpy as np
import rail1
import torch
import torch.nn.functional as F
import torchvision

import datasets
import models


def gram_schmidt(v):
    u = v.clone()
    for i in range(1, u.shape[1]):
        z = torch.einsum('bij, bj->bi', u[:, :i], v[:, i]) / torch.einsum('bij, bij->bi', u[:, :i], u[:, :i]) 
        u[:, i] = v[:, i] - torch.einsum('bi, bij->bj', z, u[:, :i])
    u /= u.norm(dim=2, keepdim=True)
    return u


def forward_and_loss_fn(batch, model):

    basis = gram_schmidt(batch[:, :3])
    batch = torch.cat([basis, batch], dim=1)

    dot = torch.einsum("bnd,bmd->bnm", batch, batch)
    input = dot.reshape(len(dot), -1)

    preds = model.forward(input, basis)
    targets = batch[:, 3:]

    loss = F.mse_loss(preds, targets, reduction="none").mean((1, 2))

    return loss.mean(0), {
        "loss": loss,
    }


@torch.no_grad()
def eval_batch(batch, batch_idx, outputs, *, validation=False):
    # if batch_idx > 0:
    return

    # predictions = outputs["predictions"].view(-1, 1, 28, 28)
    # targets = outputs["targets"].view(-1, 1, 28, 28)
    # images = torch.cat([predictions, targets], dim=-1)

    # img_grid = plot_images(images)
    # outputs["img_grid"] = img_grid

    # return outputs


def main(config):
    run_dir = config["run_dir"]
    dataset_config = config["dataset"]
    data = getattr(datasets, dataset_config.pop("name"))(**dataset_config)
    model = getattr(models, config["model"].pop("name"))(
        n_points=dataset_config["n_points"], **config["model"]
    )
    optimizer = getattr(rail1.optimizers, config["optimizer"].pop("name"))(
        model, **config["optimizer"]
    )

    device = config["device"]

    model = model.to(device)

    logging_fn = rail1.utils.get_logging_fn(
        logger=rail1.loggers.WANDBLogger() if config["wandb"] else None
    )
    # metric_fns = [
    #     functools.partial(rail1.metrics.mean_key, key="loss"),
    #     functools.partial(rail1.metrics.figure_key, key="img_grid"),
    # ]

    rail1.fit(
        run_dir,
        model,
        optimizer,
        data,
        forward_and_loss_fn=forward_and_loss_fn,
        logging_fn=logging_fn,
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
