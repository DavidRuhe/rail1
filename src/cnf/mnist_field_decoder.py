import torchvision
import torch
import torch.nn.functional as F
import rail1
import models
import functools
import datasets


def sample_random_locations(image_tensor, n):
    batch_size = len(image_tensor)
    coordinates = torch.rand(batch_size, n, 2, device="cuda") * 2 - 1
    targets = F.grid_sample(
        image_tensor, coordinates.unsqueeze(2), align_corners=True, mode="nearest"
    )
    return coordinates, targets


def forward_and_loss_fn(batch, model, *, num_locs_per_sample=512):
    images, _, z = batch

    coordinates, targets = sample_random_locations(images, num_locs_per_sample)
    z = z[:, None].expand(-1, num_locs_per_sample, -1)
    preds = model(coordinates, z)

    targets = targets.squeeze(1)

    loss = F.mse_loss(preds, targets, reduction="none")
    loss = loss.mean((1, 2))
    return loss.mean(0), {
        "loss": loss,
        "predictions": preds,
        "targets": images,
    }


def render_field(model, z, grid_size=28):

    linspace = torch.linspace(-1, 1, grid_size, device=z.device)
    x, y = torch.meshgrid(linspace, linspace, indexing="xy")
    coordinates = torch.stack([x, y], dim=-1).view(1, -1, 2)

    z = z[:, None].expand(-1, grid_size * grid_size, -1)
    preds = model(coordinates, z)
    preds = preds.view(-1, 1, grid_size, grid_size)
    return preds


@torch.no_grad()
def eval_batch(batch, batch_idx, outputs, *, model, validation=False):
    if batch_idx > 0:
        return

    embeddings = batch[2]

    predictions = render_field(model, embeddings)
    targets = batch[0]
    images = torch.cat([predictions, targets], dim=-1)

    img_grid = rail1.utils.plot_images(images)
    outputs["img_grid"] = img_grid

    return outputs


def main(config):
    run_dir = config["run_dir"]
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
        eval_batch_fn=functools.partial(eval_batch, model=model),
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
