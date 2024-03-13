import functools

import matplotlib.pyplot as plt
import numpy as np
import rail1
import torch
import torch.nn.functional as F
import torchvision
import plotly.graph_objects as go


import datasets
import models


def forward_and_loss_fn(points, model):

    positive = torch.ones(points.shape[:2], dtype=points.dtype, device=points.device)
    negative = torch.zeros(points.shape[:2], dtype=points.dtype, device=points.device)
    negative_points = torch.randn_like(points) * 2 - 1

    input = torch.cat([points, negative_points], dim=1)
    targets = torch.cat([positive, negative], dim=1)

    # input = torch.einsum("bnd,bmd->bnm", input, input).reshape(len(input), -1)

    input = input.view(-1, 3)

    preds = model.forward(input)

    targets = targets.view(-1, 1)

    loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none").sum(-1)

    return loss.mean(0), {
        "loss": loss,
    }


import plotly.graph_objects as go


def plotly_volume(values):
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()

    assert values.ndim == 3

    x, y, z = np.indices(np.array(values.shape) + 1) - 0.5

    breakpoint()

    vol = go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=0.1,
        isomax=0.9,
        opacity=0.1,
        surface_count=20,
    )

    return go.Figure(data=vol)


@torch.no_grad()
def eval_batch(points, batch_idx, outputs, *, model, validation=False):
    if batch_idx > 0:
        return

    outputs["points"] = points[0]

    # 3d grid
    # linspace = torch.linspace(-1, 1, 128, device=points.device, dtype=points.dtype)
    # grid = torch.stack(
    #     torch.meshgrid(linspace, linspace, linspace, indexing="ij"), dim=-1
    # )
    # grid = grid.reshape(-1, 3)

    X, Y, Z = np.mgrid[-1:1:32j, -1:1:32j, -1:1:32j]
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    grid = torch.from_numpy(grid).to(points.device).float()
    is_surface = model.forward(grid).sigmoid().cpu().numpy().flatten()

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=is_surface,
            isomin=0.1,
            isomax=0.9,
            opacity=0.1,
            surface_count=20,
        )
    )
    outputs['volume'] = fig
    return outputs

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
        n_points=dataset_config["n_points_per_shape"], **config["model"]
    )
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
        functools.partial(rail1.metrics.figure_key, key="points"),
        functools.partial(rail1.metrics.figure_key, key="volume"),
    ]

    rail1.fit(
        run_dir,
        model,
        optimizer,
        data,
        forward_and_loss_fn=forward_and_loss_fn,
        logging_fn=logging_fn,
        eval_batch_fn=functools.partial(eval_batch, model=model),
        metrics_fns=metric_fns,
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
