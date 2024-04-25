import functools

import matplotlib.pyplot as plt
import numpy as np
import rail1
import torch
import torch.nn.functional as F
import torchvision

import datasets
import models



def forward_and_loss_fn(batch, model):
    
    x, rad, label = batch

    rad = rad[:, None].float()

    label = F.one_hot(label, 5).float()


    x_pos = x
    x_neg = torch.rand_like(x_pos) * 2 - 1

    label_pos = torch.ones(*x_pos.shape[:-1], device=x_pos.device)
    label_neg = torch.zeros(*x_pos.shape[:-1],device=x_pos.device)

    x = torch.cat([x_pos, x_neg], dim=0)
    rad = rad.repeat(2, 1)
    label = label.repeat(2, 1)

    logits = model(x, rad, label).squeeze(-1)
    labels = torch.cat([label_pos, label_neg], dim=0)
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    return loss, {"logits": logits, "targets": labels}


@torch.no_grad()
def eval_batch(points, batch_idx, outputs, *, model, validation=False):
    if batch_idx > 0:
        return
    
    x_pos, rad, label = points

    x_pos = x_pos[:1]
    rad = rad[:1]
    label = label[:1]

    outputs["points"] = x_pos[0]

    # 3d grid
    linspace = torch.linspace(-1, 1, 64, device=x_pos.device, dtype=x_pos.dtype)
    grid = torch.stack(
        torch.meshgrid(linspace, linspace, linspace, indexing="ij"), dim=-1
    )
    grid = grid.reshape(-1, 3)
    queries = grid[None]

    label = F.one_hot(label, 5).float()
    rad = rad[:, None].float()

    logits = model(queries, rad, label).squeeze(-1)

    logits = logits.squeeze(0).squeeze(-1)

    keep = grid[logits > 0]

    outputs['points_surface'] = keep


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
        functools.partial(rail1.metrics.figure_key, key="points_surface"),
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
