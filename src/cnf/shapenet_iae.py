import torchvision
import torch
import torch.nn.functional as F
import rail1
import models
import functools
import datasets
import numpy as np


def forward_and_loss_fn(data, model):

    p = data.get("points")
    df = data.get("points.df")
    inputs = data.get("inputs", torch.empty(p.size(0), 0))

    c = model.encode_inputs(inputs)

    kwargs = {}
    # General points
    output = model.decode(p, c, **kwargs)
    loss = F.l1_loss(output, df, reduction="none").sum(-1)

    return loss.mean(0), {
        "loss": loss,
    }


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

    points = data.get("points")
    df = data.get("points.df")

    inputs = data.get("inputs", torch.empty(points.size(0), 0))

    points_iou = data.get("points_iou")
    df_iou = data.get("points_iou.df")

    batch_size = points.size(0)

    kwargs = {}

    device = points.device

    # add pre-computed index
    inputs = add_key(inputs, data.get("inputs.ind"), "points", "index", device=device)
    # add pre-computed normalized coordinates
    points = add_key(points, data.get("points.normalized"), "p", "p_n", device=device)
    points_iou = add_key(
        points_iou, data.get("points_iou.normalized"), "p", "p_n", device=device
    )

    # Compute iou
    with torch.no_grad():
        p_out = model(points_iou, inputs, **kwargs)

    df_iou_np = (df_iou >= -0.1)
    df_iou_hat_np = (p_out >= -0.1)

    iou = compute_iou(df_iou_np, df_iou_hat_np)

    outputs["neg_iou"] = -iou

    return outputs


def main(config):
    run_dir = config["run_dir"]

    # train_dataset = config.get_dataset('train', cfg)
    # val_dataset = config.get_dataset('val', cfg, return_idx=True)

    dataset_config = config["dataset"]
    data = getattr(datasets, dataset_config.pop("name"))(
        method=config["model"]["name"], **dataset_config
    )

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
        functools.partial(rail1.metrics.mean_key, key="neg_iou"),
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
