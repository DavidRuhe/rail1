import torchvision
import torch
import torch.nn.functional as F
import rail1
import datasets
import models
import functools
import matplotlib.pyplot as plt
import numpy as np


def forward_and_loss_fn(batch, model):
    images, labels = batch
    preds = model((None, images))

    preds = preds.squeeze(dim=-1)

    loss = F.cross_entropy(preds, labels, reduction="none")

    return loss.mean(0), {
        "loss": loss,
        "logits": preds,
        "targets": labels,
    }


def accuracy(metric_dicts, is_training):
    logits = torch.cat(metric_dicts["logits"])
    targets = torch.cat(metric_dicts["targets"])
    preds = logits.argmax(dim=-1)
    return {"accuracy": (preds == targets).float().mean()}


def mean_key(metric_dicts, key):
    return {key: torch.mean(torch.cat(metric_dicts[key]))}


def mean_loss(metric_dicts, is_training):
    return mean_key(metric_dicts, "loss")


def figure(metric_dicts, is_training):
    if "attention_maps" not in metric_dicts:
        return {}
    return {"attention_maps": metric_dicts["attention_maps"].pop()}


def img_grid(metric_dicts, is_training):
    if "img_grid" not in metric_dicts:
        return {}
    return {"img_grid": metric_dicts["img_grid"].pop()}


def plot_images(imgs):
    DATA_MEANS = np.array([0.485, 0.456, 0.406])
    DATA_STD = np.array([0.229, 0.224, 0.225])

    # # As torch tensors for later preprocessing
    TORCH_DATA_MEANS = torch.from_numpy(DATA_MEANS).view(1, 3, 1, 1)
    TORCH_DATA_STD = torch.from_numpy(DATA_STD).view(1, 3, 1, 1)

    images = torch.stack(imgs)
    images = images * TORCH_DATA_STD + TORCH_DATA_MEANS

    img_grid = torchvision.utils.make_grid(
        images, nrow=10, normalize=True, pad_value=0.5, padding=16
    )
    img_grid = img_grid.permute(1, 2, 0)

    return img_grid


# @torch.no_grad()
# def eval_batch(
#     batch, batch_idx, outputs, model, train_dataset, test_dataset, validation=False
# ):
#     if batch_idx > 0:
#         return

#     inp_data = outputs["inp_data"]
#     indices = outputs["indices"]
#     attention_maps = get_attention_maps(model, inp_data)

#     idx = 0

#     indices = indices[idx].detach().cpu()
#     if validation:
#         imgs = [train_dataset[i][0] for i in indices]
#     else:
#         imgs = [test_dataset[i][0] for i in indices]

#     img_grid = plot_images(imgs)
#     fig = plot_attention_maps(None, attention_maps, idx=0)

#     outputs["attention_maps"] = fig
#     outputs["img_grid"] = img_grid

#     return outputs


def module_to_device(module, device):
    if isinstance(module, torch.nn.Module):
        module.to(device)
    elif isinstance(module, (tuple, list)):
        for m in module:
            module_to_device(m, device)
    else:
        raise ValueError(f"Unknown type {type(module)}.")


def main(config):
    run_dir = config["run_dir"]
    dataset_config = config["dataset"]
    data = getattr(rail1.datasets, dataset_config.pop("name"))(**dataset_config)
    model = getattr(models, config["model"].pop("name"))(**config["model"])
    optimizer = getattr(rail1.optimizers, config["optimizer"].pop("name"))(
        model, **config["optimizer"]
    )

    device = config["device"]
    module_to_device(model, device)

    forward_and_loss_fn_ = functools.partial(forward_and_loss_fn)

    logging_fn_ = functools.partial(
        rail1.loggers.default_log_fn,
        logger=rail1.loggers.WANDBLogger() if config["wandb"] else None,
    )

    # # eval_batch_fn_ = functools.partial(
    # #     eval_batch,
    # #     model=model,
    # #     train_dataset=data["cifar100_train"],
    # #     test_dataset=data["cifar100_test"],
    # # )
    # eval_batch_fn_ = None

    scheduler = rail1.schedulers.CosineAnnealingLR(optimizer, **config["scheduler"])

    rail1.fit(
        run_dir,
        model,
        optimizer,
        data,
        forward_and_loss_fn=forward_and_loss_fn_,
        metrics_fns=[mean_loss, accuracy, figure, img_grid],
        logging_fn=logging_fn_,
        scheduler=scheduler,
        # eval_batch_fn=eval_batch_fn_,
        eval_batch_fn=None,
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
