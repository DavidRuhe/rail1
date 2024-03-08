import functools

import numpy as np
import rail1
import torch
import torch.nn.functional as F
import torchvision

import datasets
import models


def forward_and_loss_fn(data, model):

    points, labels, surface = data
    # p = data.get("points")
    # df = data.get("points.df")
    # inputs = data.get("inputs", torch.empty(p.size(0), 0))

    model_outputs = model(surface, points)
    if "kl" in model_outputs:
        loss_kl = model_outputs["kl"]
        # loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
    else:
        loss_kl = None

    model_outputs = model_outputs["logits"]

    loss_vol = F.binary_cross_entropy_with_logits(
        model_outputs[:, :1024], labels[:, :1024], reduction="none"
    )
    loss_near = F.binary_cross_entropy_with_logits(
        model_outputs[:, 1024:], labels[:, 1024:], reduction="none"
    )

    loss_vol = loss_vol.mean(1)
    loss_near = loss_near.mean(1)

    backprop_loss = loss_vol + 0.1 * loss_near

    if loss_kl is not None:
        backprop_loss += 1e-3 * loss_kl

    return backprop_loss.mean(0), {"loss": backprop_loss}


def add_key(base, new, base_name, new_name, device=None):
    """Add new keys to the given input

    Args:
        base (tensor): inputs
        new (tensor): new info for the inputs
        base_name (str): name for the input
        new_name (str): name for the new info
        device (device): pytorch device
    """
    if (new is not None) and (isinstance(new, dict)):
        if device is not None:
            for key in new.keys():
                new[key] = new[key].to(device)
        base = {base_name: base, new_name: new}
    return base


def compute_iou(occ1, occ2):
    """Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    """
    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = occ1 >= 0.5
    occ2 = occ2 >= 0.5

    # Compute IOU
    area_union = (occ1 | occ2).float().sum(axis=-1)
    area_intersect = (occ1 & occ2).float().sum(axis=-1)
    iou = area_intersect / area_union

    if area_union == 0:
        return area_union

    return iou


@torch.no_grad()
def eval_batch(data, batch_idx, outputs, *, model, validation=False):

    points, labels, surface = data

    logits = model(surface, points)['logits']
    threshold = 0

    pred = torch.zeros_like(logits)
    pred[logits>=threshold] = 1

    accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
    intersection = (pred * labels).sum(dim=1)
    union = (pred + labels).gt(0).sum(dim=1)
    iou = intersection * 1.0 / union + 1e-5

    outputs['inv_iou'] = 1 / iou
    outputs['inv_accuracy'] = 1 / accuracy

    # Compute loss
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = loss.mean(1)

    outputs["loss"] = loss

    return outputs


def main(config):
    run_dir = config["run_dir"]

    # train_dataset = config.get_dataset('train', cfg)
    # val_dataset = config.get_dataset('val', cfg, return_idx=True)

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
    # scheduler = rail1.schedulers.CosineAnnealingLR(optimizer, **config["scheduler"])
    scheduler = None

    metric_fns = [
        functools.partial(rail1.metrics.mean_key, key="loss"),
        functools.partial(rail1.metrics.mean_key, key="inv_iou"),
        functools.partial(rail1.metrics.mean_key, key="inv_accuracy"),
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
        eval_batch_fn=functools.partial(eval_batch, model=model),
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
