import functools

import plotly.graph_objects as go
import rail1
import torch
import torch.nn.functional as F

import datasets
import models


def forward_and_loss_fn(input, model):

    points_idx, labels = input
    preds = model.forward(*points_idx)
    labels = labels.squeeze(-1)

    loss = F.cross_entropy(preds, labels, reduction="none")

    return loss.mean(0), {
        "loss": loss,
        "logits": preds,
        "targets": labels,
    }




@torch.no_grad()
def eval_batch(points, batch_idx, outputs, *, model, idx_to_label, validation=False):
    if batch_idx > 0:
        return


def main(config):
    run_dir = config["run_dir"]
    dataset_config = config["dataset"]
    data = getattr(datasets, dataset_config.pop("name"))(**dataset_config)
    model = getattr(models, config["model"].pop("name"))(**config["model"])
    optimizer = getattr(rail1.optimizers, config["optimizer"].pop("name"))(
        model, **config["optimizer"]
    )
    if "scheduler" in config:
        scheduler = getattr(rail1.schedulers, config["scheduler"].pop("name"))(
            optimizer, **config["scheduler"]
        )
    else:
        scheduler = None

    device = config["device"]

    model = model.to(device)

    logging_fn = rail1.utils.get_logging_fn(
        logger=rail1.loggers.WANDBLogger() if config["wandb"] else None
    )
    metric_fns = [
        functools.partial(rail1.metrics.mean_key, key="loss"),
        functools.partial(rail1.metrics.figure_key, key="points"),
        functools.partial(rail1.metrics.figure_key, key="plotly"),
        rail1.metrics.accuracy,
        # functools.partial(rail1.metrics.figure_key, key="volume"),
    ]

    rail1.fit(
        run_dir,
        model,
        optimizer,
        data,
        scheduler=scheduler,
        forward_and_loss_fn=forward_and_loss_fn,
        logging_fn=logging_fn,
        eval_batch_fn=functools.partial(
            eval_batch, model=model, idx_to_label=data["idx_to_label"]
        ),
        metrics_fns=metric_fns,
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
