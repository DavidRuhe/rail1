import os

import torch
import wandb
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def default_log_fn(metrics, step, *, logger):
    if logger is not None:
        logger.log_all(metrics, step)


class WANDBLogger:
    def __init__(self):
        if torch.distributed.is_initialized():
            assert (
                torch.distributed.get_rank() == 0
            ), "WANDBLogger should only be initialized on rank 0."

        self.metrics = set()
        self.dir = wandb.run.dir

    @property
    def initialized(self):
        return wandb.run is not None

    def _log(self, dict, step):
        if not self.initialized:
            return
        wandb.log(dict, step=step)

    def log_metrics(self, metrics, step):
        if not self.initialized:
            return

        for m in metrics:
            if m not in self.metrics:
                wandb.define_metric(m, summary="max,min,last")
                print(f"Defined metric {m}.")
                self.metrics.add(m)

        return self._log(metrics, step)

    def log_image(self, image_dict, step):
        image_dict = {k: wandb.Image(v) for k, v in image_dict.items()}
        return self._log(image_dict, step)

    def log_3d_object(self, object_dict, step):
        object_dict = {k: wandb.Object3D(v) for k, v in object_dict.items()}
        return self._log(object_dict, step)
    
    def log_plotly(self, plotly_dict, step):
        plotly_dict = {k: wandb.Plotly(v) for k, v in plotly_dict.items()}
        return self._log(plotly_dict, step)

    def log_all(self, dict, step):
        for k, v in dict.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    self.log_metrics({k: v}, step)
                elif v.ndim == 2 and v.shape[-1] == 3:
                    self.log_3d_object({k: v.cpu().numpy()}, step)
                elif v.ndim == 3:
                    self.log_image({k: v.cpu().numpy()}, step)
                else:
                    raise ValueError(f"Can't log tensor {k} of shape {v.shape}.")
            elif isinstance(v, (float, int)):
                self.log_metrics({k: v}, step)
            elif isinstance(v, plt.Figure):
                self.log_image({k: v}, step)
                plt.close(v)
            elif go is not None and isinstance(v, go.Figure):
                self.log_plotly({k: v}, step)
            else:
                raise ValueError(f"Can't log {k} of type {type(v)}.")

    def save(self, file):
        if not self.initialized:
            print("Not saving because WANDB is not initialized.")
            return
        wandb.save(file, base_path=os.path.dirname(file))


def _pp(d, indent=0):
    for key, value in d.items():
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            _pp(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))


class ConsoleLogger:
    def __init__(self) -> None:
        self.metrics = []
        self.dir = None

    def _log(self, dict, step):
        # Print metrics
        print()
        for k, v in dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            print(f"{k}: {v:.4f}")
        print()

    def log_metrics(self, metrics, step):
        for m in metrics:
            if m not in self.metrics:   
                print(f"Defined metric {m}.")
                self.metrics.append(m)

        return self._log(metrics, step)
