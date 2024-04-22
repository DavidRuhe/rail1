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
    
    x, rad = batch

    rad = rad[:, None].float()

    x_pos = x
    x_neg = torch.rand_like(x_pos) * 2 - 1

    label_pos = torch.ones(*x_pos.shape[:-1], device=x_pos.device)
    label_neg = torch.zeros(*x_pos.shape[:-1],device=x_pos.device)

    z = torch.cat([rad, rad], dim=0)
    x = torch.cat([x_pos, x_neg], dim=0)

    logits = model(x, z).squeeze(-1)
    # logits = model(x).squeeze(-1)
    labels = torch.cat([label_pos, label_neg], dim=0)
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    accuracy = ((logits > 0).float() == labels).float().mean()

    print(accuracy)

    return loss, {}


@torch.no_grad()
def eval_batch(batch, batch_idx, outputs, *, validation=False):
    return

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
