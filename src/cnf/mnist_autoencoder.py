import torchvision
import torch
import torch.nn.functional as F
import rail1
import models
import functools
import matplotlib.pyplot as plt
import numpy as np


def forward_and_loss_fn(batch, model):
    images, _ = batch
    images = images.view(len(images), -1)
    preds = model.forward(images)

    loss = F.mse_loss(preds, images, reduction="none").mean(1)

    return loss.mean(0), {
        "loss": loss,
        "predictions": preds,
        "targets": images,
    }


def plot_images(imgs, max_number=256):
    imgs = imgs[:max_number]

    img_grid = torchvision.utils.make_grid(
        imgs, nrow=int(len(imgs) ** 0.5), normalize=True, pad_value=0.5
    )
    img_grid = img_grid.permute(1, 2, 0)

    return img_grid


@torch.no_grad()
def eval_batch(batch, batch_idx, outputs, *, validation=False):
    if batch_idx > 0:
        return

    predictions = outputs["predictions"].view(-1, 1, 28, 28)
    targets = outputs["targets"].view(-1, 1, 28, 28)
    images = torch.cat([predictions, targets], dim=-1)

    img_grid = plot_images(images)
    outputs["img_grid"] = img_grid

    return outputs


def main(config):
    run_dir = config["run_dir"]
    dataset_config = config["dataset"]
    data = getattr(rail1.datasets, dataset_config.pop("name"))(**dataset_config)
    model = getattr(models, config["model"].pop("name"))(**config["model"])
    optimizer = getattr(rail1.optimizers, config["optimizer"].pop("name"))(
        model, **config["optimizer"]
    )

    device = config["device"]

    model = model.to(device)

    logging_fn = rail1.utils.get_logging_fn(
        logger=rail1.loggers.WANDBLogger() if config["wandb"] else None
    )
    scheduler = rail1.schedulers.CosineAnnealingLR(optimizer, **config["scheduler"])

    metric_fns = [
        functools.partial(rail1.metrics.mean_key, key="loss"),
        functools.partial(rail1.metrics.figure_key, key="img_grid"),
    ]
    
    rail1.fit(
        run_dir,
        model,
        optimizer,
        data,
        forward_and_loss_fn=forward_and_loss_fn,
        metrics_fns=metric_fns,
        logging_fn=logging_fn,
        scheduler=scheduler,
        eval_batch_fn=functools.partial(eval_batch),
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
