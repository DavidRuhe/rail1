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


def forward_and_loss_fn(input, model):
    
    points, labels = input
    preds = model.forward(points)

    loss = F.cross_entropy(preds, labels, reduction="none")

    return loss.mean(0), {
        "loss": loss,
        "logits": preds,
        "targets": labels,
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

    points, labels = points
    outputs['points'] = points[0]
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
    model = getattr(models, config["model"].pop("name"))(**config["model"]
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
        rail1.metrics.inv_accuracy
        # functools.partial(rail1.metrics.figure_key, key="volume"),
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
