import os

import random
import numpy
import torch
import torch.distributed
import wandb


# class Checkpoint:
#     def __init__(self, metrics=None, dir=None):
#         super().__init__()

#         self.dir = dir
#         self._cached_model_state_dict = None
#         self._cached_optimizer_state_dict = None
#         self._cached_epoch = None
#         self._cached_step = None

#         if dir is not None:
#             metrics = self.load_checkpoint(dir)

#         if type(metrics) == str:
#             metrics = (metrics,)
#         if type(metrics) in (list, tuple):
#             metrics = {m: float("inf") for m in metrics}

#         self.best_metrics = metrics

#         self.save_paths = {}

#     def load_checkpoint(self, dir):
#         state_dict = torch.load(dir)
#         model = state_dict["model"]
#         optimizer = state_dict["optimizer"]
#         metrics = state_dict["metrics"]
#         epoch = state_dict["epoch"]
#         step = state_dict["step"]
#         self._cached_model_state_dict = model
#         self._cached_optimizer_state_dict = optimizer
#         self._cached_epoch = epoch
#         self._cached_step = step
#         return metrics

#     def restore(self, trainer, model, optimizer):
#         if self._cached_model_state_dict is not None:
#             if torch.distributed.is_initialized():
#                 model.module.load_state_dict(self._cached_model_state_dict)
#             else:
#                 model.load_state_dict(self._cached_model_state_dict)
#             print(f"Successfully restored model state dict from {self.dir}!")
#         if self._cached_optimizer_state_dict is not None:
#             optimizer.load_state_dict(self._cached_optimizer_state_dict)
#             print(f"Successfully restored optimizer state dict from {self.dir}!")

#         if self._cached_epoch is not None:
#             trainer.current_epoch = self._cached_epoch
#             print(f"Set current epoch to {self._cached_epoch}.")

#         if self._cached_step is not None:
#             trainer.global_step = self._cached_step
#             print(f"Set global step to {self._cached_step}.")

#         self._cached_epoch = None
#         self._cached_step = None
#         self._cached_model_state_dict = None
#         self._cached_optimizer_state_dict = None

#     @property
#     def _is_master(self):
#         if torch.distributed.is_initialized():
#             return torch.distributed.get_rank() == 0
#         else:
#             return True

#     def on_test_end(self, trainer, model, optimizer, metrics, *args, **kwargs):
#         # if trainer.logger is None:
#         #     print(f"No logger found, skipping checkpoint.")
#         #     return

#         # if trainer.logger.dir is None:
#         #     print("Logger has no directory, skipping checkpoint.")
#         #     return

#         should_write = (
#             self._is_master
#             and trainer.logger is not None
#             and trainer.logger.dir is not None
#         )

#         epoch = trainer.current_epoch
#         step = trainer.global_step

#         for m, v in self.best_metrics.items():

#             if metrics[m] < v:
#                 self.best_metrics[m] = metrics[m]
#                 # save_path = os.path.join(
#                 #     dir,
#                 #     f"epoch_{epoch}_step_{step}_{m.replace('/', '_')}={metrics[m]:.4f}.pt",
#                 # )

#                 model_state_dict = (
#                     model.module.state_dict()
#                     if torch.distributed.is_initialized()
#                     else model.state_dict()
#                 )
#                 checkpoint = {
#                     "model": model_state_dict,
#                     "optimizer": optimizer.state_dict(),
#                     "metrics": self.best_metrics,
#                     "epoch": epoch,
#                     "step": step,
#                 }

#                 if should_write:
#                     alias = f"best_{m.replace('/', '_')}"
#                     save_path = os.path.join(
#                         trainer.logger.dir,
#                         alias,
#                     )

#                     torch.save(checkpoint, save_path)
#                     trainer.logger.save_model(save_path, alias=alias)

#                     if m in self.save_paths:
#                         os.remove(self.save_paths[m])
#                     self.save_paths[m] = save_path

#                     print(
#                         f"Metric {m} improved to {metrics[m]:.4f}, saving checkpoint. Saved checkpoint to {save_path}. Initializing test loop."
#                     )
#                 trainer.should_test = True


def save_wandb(file, metadata=None):

    # Method 1
    # if wandb.run is not None:
    #     wandb.save(file, base_path = split_path(file, 2))

    # Method 2
    name = str(wandb.run.id) + "-" + "checkpoint"
    artifact = wandb.Artifact(name, type="checkpoint", metadata=metadata)
    artifact.add_file(file)
    wandb.log_artifact(artifact)

    # Remove old artifacts
    # project = wandb.run.project
    # entity = wandb.run.entity
    # id = wandb.run.id
    # run = wandb.Api().run(f"{entity}/{project}/{id}")
    # for v in run.logged_artifacts():
    #         if len(v.aliases) == 0:
    #             v.delete()


def save_checkpoint(checkpoint_dir, model, train_state, optimizer, metrics=None):
    # if trainer.logger is None:
    #     print(f"No logger found, skipping checkpoint.")
    #     return

    # if trainer.logger.dir is None:
    #     print("Logger has no directory, skipping checkpoint.")
    #     return
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    is_distributed = torch.distributed.is_initialized()
    if is_distributed:
        raise NotImplementedError("Should support multiple random states.")
    should_write = torch.distributed.get_rank() == 0 if is_distributed else True

    model_state_dict = (
        model.module.state_dict()
        if torch.distributed.is_initialized()
        else model.state_dict()
    )

    random_state = {
        "torch": torch.get_rng_state(),
        "numpy": numpy.random.get_state(),
        "random": random.getstate(),
        "cuda": torch.cuda.get_rng_state(),
        "cuda_all": torch.cuda.get_rng_state_all(),
    }

    checkpoint = {
        "model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "train_state": train_state,
        "random_state": random_state,
    }

    def get_scalar(v):
        if isinstance(v, (float, int)):
            return v
        elif isinstance(v, torch.Tensor) and v.dim() == 0:
            return v.cpu().item()

    scalar_metrics = None
    if metrics is not None:
        scalar_metrics = {
            k: get_scalar(v) for k, v in metrics.items() if get_scalar(v) is not None
        }
        assert all(
            v >= 0 for v in scalar_metrics.values()
        ), "Only non-negative metrics are supported."
        metrics_str = "-".join([f"{k}={v:.4f}" for k, v in scalar_metrics.items()])
        metrics_str = metrics_str.replace("/", "_")
        filename = os.path.join(
            checkpoint_dir,
            f"step={train_state['global_step']}-epoch={train_state['current_epoch']}-{metrics_str}.pt",
        )
    else:
        filename = os.path.join(
            checkpoint_dir,
            f"step={train_state['global_step']}-epoch={train_state['current_epoch']}.pt",
        )

    if should_write:
        torch.save(checkpoint, filename)
        if wandb.run is not None:
            save_wandb(filename, metadata={"filename": filename})
            os.remove(filename)
        print(f"Successfully saved checkpoint to {filename}")

    #     trainer.logger.save_model(save_path, alias=alias)

    #     if m in self.save_paths:
    #         os.remove(self.save_paths[m])
    #     self.save_paths[m] = save_path

    #     print(
    #         f"Metric {m} improved to {metrics[m]:.4f}, saving checkpoint. Saved checkpoint to {save_path}. Initializing test loop."
    #     )
    # trainer.should_test = True


def get_sorted_checkpoints(checkpoint_dir):

    checkpoints = os.listdir(checkpoint_dir)
    try:
        steps = [int(c.split("-")[0].split("=")[1]) for c in checkpoints]
    except IndexError as e:  # pragma: no cover
        print(f"Could not process checkpoints {checkpoints}")
        raise e
    checkpoints.sort(key=dict(zip(checkpoints, steps)).get)
    return checkpoints


def load_checkpoint(checkpoint_dir, model, train_state, optimizer):

    is_distributed = torch.distributed.is_initialized()
    if is_distributed:  # pragma: no cover
        raise NotImplementedError("Should support multiple random states.")

    checkpoints = get_sorted_checkpoints(checkpoint_dir)
    checkpoint = checkpoints[-1]

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)

    state_dict = torch.load(checkpoint_path)

    model_state_dict = state_dict["model"]
    optimizer_state_dict = state_dict["optimizer"]
    train_state_dict = state_dict["train_state"]
    random_state_dict = state_dict["random_state"]

    model.load_state_dict(model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    train_state.update(train_state_dict)

    torch.set_rng_state(random_state_dict["torch"])
    torch.cuda.set_rng_state(random_state_dict["cuda"])
    torch.cuda.set_rng_state_all(random_state_dict["cuda_all"])
    numpy.random.set_state(random_state_dict["numpy"])
    random.setstate(random_state_dict["random"])

    print(f"\nSuccessfully restored complete state from: {checkpoint_path}\n")

    return train_state
