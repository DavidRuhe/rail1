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
    idx0 = torch.randint(0, x.shape[1], (x.shape[0], x.shape[1]))
    idx1 = torch.randint(0, idx0.shape[1], (x.shape[0], idx0.shape[1] // 2))
    idx2 = torch.randint(0, idx1.shape[1], (x.shape[0], idx1.shape[1] // 2))
    logits = model(x, (idx0, idx1, idx2))

    rad = rad.float()

    loss = F.mse_loss(logits.squeeze(-1), rad)
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
