import torch
import models
import torch.nn.functional as F
import rail1
from rail1 import fire
from rail1.training import fit


def forward_and_loss_fn(batch, model):
    images, labels = batch
    logits = model(images)
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss.mean(0), {"loss": loss}


def mean_key(metric_dicts, key):
    return {key: torch.mean(torch.cat(metric_dicts[key]))}


def mean_loss(metric_dicts):
    return mean_key(metric_dicts, "loss")


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

    fit.fit(
        run_dir,
        model,
        optimizer,
        data,
        forward_and_loss_fn=forward_and_loss_fn,
        metrics_fns=(mean_loss,),
        log_metrics_fn=None,
        **config["fit"],
    )

    # if config["dist"] is not None:
    #     local_rank = config["dist"]["local_rank"]
    #     device = torch.device(f"cuda:{local_rank}")
    #     model = model.to(device)
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model = DistributedDataParallel(model)
    # else:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = model.to(device)
    # print(f"Using device: {device}")

    # optimizer_config = config["optimizer"]
    # optimizer = rail1.load_module(optimizer_config.pop("module"))(
    #     model.parameters(), **optimizer_config
    # )
    # steps = config["trainer"]["max_steps"]
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     steps,
    #     warmup_steps=int(6e3),
    #     decay_steps=int(1 / 3 * steps),
    # )

    # trainer_module = rail1.load_module(config["trainer"].pop("module"))

    # trainer_config = config["trainer"]
    # trainer_config["scheduler"] = scheduler
    # trainer_config["wandb"] = config["wandb"]
    # trainer = trainer_module(
    #     **trainer_config,
    # )
    # trainer.fit(model, optimizer, train_loader, val_loader, test_loader)


def test():
    batch = torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,))
    model = lambda x: torch.randn(2, 10)
    loss, metrics = forward_and_loss_fn(batch, model)


if __name__ == "__main__":
    fire.fire(main)
