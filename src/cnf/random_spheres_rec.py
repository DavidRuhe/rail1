import functools

import matplotlib.pyplot as plt
import numpy as np
import rail1
import torch
import torch.nn.functional as F
import torchvision

import datasets
import models
import plotly.graph_objects as go



def forward_and_loss_fn(batch, model):

    
    x_pos, radius = batch
    # idx0 = torch.randint(0, x_pos.shape[1], (x_pos.shape[0], x_pos.shape[1]))
    # idx1 = torch.randint(0, idx0.shape[1], (x_pos.shape[0], idx0.shape[1] // 2))
    # idx2 = torch.randint(0, idx1.shape[1], (x_pos.shape[0], idx1.shape[1] // 2))
    label_pos = torch.ones(*x_pos.shape[:-1], device=x_pos.device)
    x_neg = torch.rand_like(x_pos) * 2 - 1
    label_neg = torch.zeros(*x_pos.shape[:-1],device=x_pos.device)

    pc = x_pos
    queries = torch.cat([x_pos, x_neg], dim=0)
    pc = pc.repeat(2, 1, 1)

    logits = model(pc, queries)['logits']



    labels = torch.cat([label_pos, label_neg], dim=0)
    # logits = model(x_pos, x_neg, (idx0, idx1, idx2)).squeeze(-1)


    loss = F.binary_cross_entropy_with_logits(logits, labels)

    return loss, {"logits": logits, "targets": labels}


def plotly_volume(values):
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()

    assert values.ndim == 3

    x, y, z = np.indices(np.array(values.shape) + 1) - 0.5

    print(values.max())

    vol = go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=0,
        isomax=1,
        opacity=0.1,
        surface_count=20,
    )

    return go.Figure(data=vol)
    




@torch.no_grad()
def eval_batch(points, batch_idx, outputs, *, model, validation=False):
    if batch_idx > 0:
        return
    
    points, radius = points

    outputs["points"] = points[0]

    # 3d grid
    linspace = torch.linspace(-1, 1, 64, device=points.device, dtype=points.dtype)
    grid = torch.stack(
        torch.meshgrid(linspace, linspace, linspace, indexing="ij"), dim=-1
    )
    grid = grid.reshape(-1, 3)

    is_surface = model.forward(points[:1], grid[None])

    logits = is_surface['logits']

    outputs['volume'] = plotly_volume(logits.view(64, 64, 64))


    return outputs


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
        rail1.metrics.binary_accuracy,
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
        **config["fit"],
        metrics_fns=metric_fns,
        eval_batch_fn=functools.partial(eval_batch, model=model),
    )


if __name__ == "__main__":
    rail1.fire(main)
