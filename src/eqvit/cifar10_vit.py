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
    preds = model.forward(images)
    preds = preds.squeeze(dim=-1)  # Shape: [Batch_size, set_size]
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


def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(
        num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size)
    )
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin="lower", vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")
    fig.subplots_adjust(hspace=0.5)

    return fig


@torch.no_grad()
def get_attention_maps(model, x, mask=None, add_positional_encoding=True):
    """
    Function for extracting the attention matrices of the whole Transformer for a single batch.
    Input arguments same as the forward pass.
    """
    x = model.input_net(x)
    if add_positional_encoding:
        x = model.positional_encoding(x)
    attention_maps = model.transformer.get_attention_maps(x, mask=mask)
    return attention_maps


def plot_images(imgs, is_training):
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


@torch.no_grad()
def eval_batch(
    batch, batch_idx, outputs, model, train_dataset, test_dataset, validation=False
):
    if batch_idx > 0:
        return

    inp_data = outputs["inp_data"]
    indices = outputs["indices"]
    attention_maps = get_attention_maps(model, inp_data)

    idx = 0

    indices = indices[idx].detach().cpu()
    if validation:
        imgs = [train_dataset[i][0] for i in indices]
    else:
        imgs = [test_dataset[i][0] for i in indices]

    img_grid = plot_images(imgs)
    fig = plot_attention_maps(None, attention_maps, idx=0)

    outputs["attention_maps"] = fig
    outputs["img_grid"] = img_grid

    return outputs


def logging_fn(metrics, step, logger):
    if logger is not None:
        logger.log_all(metrics, step)


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

    forward_and_loss_fn_ = functools.partial(forward_and_loss_fn)

    logging_fn_ = functools.partial(
        logging_fn,
        logger=rail1.loggers.WANDBLogger() if config["wandb"] else None,
        # model=model,
        # num_classes=dataset_config["num_classes"],
    )

    # eval_batch_fn_ = functools.partial(
    #     eval_batch,
    #     model=model,
    #     train_dataset=data["cifar100_train"],
    #     test_dataset=data["cifar100_test"],
    # )
    eval_batch_fn_ = None

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
        eval_batch_fn=eval_batch_fn_,
        **config["fit"],
    )


if __name__ == "__main__":
    rail1.fire(main)
