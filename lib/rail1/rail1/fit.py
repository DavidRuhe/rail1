import datetime
import os
import time
from collections import defaultdict

import torch
from torch import nn

from rail1.callbacks import checkpoint
from rail1.utils import math as math_utils
from rail1.utils import printing


def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def to_device(input, device, detach=True):
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list):
        keys = range(len(input))
    elif isinstance(input, dict):
        keys = input.keys()
    else:
        input = input.to(device)
        if detach:
            input = input.detach()
        return input
    for k in keys:
        input[k] = to_device(input[k], device)
    return input


def apply_metric_fns(metric_dicts, metric_fns, is_training):
    result = {}
    for fn in metric_fns:
        result.update(fn(metric_dicts, is_training=is_training))
    return result


def append_to_metrics_(result, metrics_dict, to_cpu=False):
    for k, v in result.items():
        metrics_dict[k].append(to_device(v, "cpu" if to_cpu else None))


@torch.no_grad()
def test_loop(
    train_state,
    model,
    forward_and_loss_fn,
    test_loader,
    metric_fns,
    log_metrics_fn,
    limit_batches=float("inf"),
    validation=False,
    print_interval=32,
):
    model.eval()

    num_test_batches = math_utils.ceildiv(
        len(test_loader.dataset), test_loader.batch_size
    )
    num_iterations = min(num_test_batches, limit_batches)
    assert num_iterations > 0
    t0 = time.time()

    # if self.is_distributed:
    #     assert model.module.test_metrics.empty()  # type: ignore
    # else:
    metrics = defaultdict(list)
    if validation:
        print_str = "Validation"
        prefix = "val"
    else:
        print_str = "Testing"
        prefix = "test"

    for batch_idx in range(num_iterations):
        batch = test_loader[batch_idx]
        batch = to_device(batch, train_state["device"])
        _, outputs = forward_and_loss_fn(batch, model)

        # if self.is_distributed:
        #     model.module.test_metrics.update(**outputs)  # type: ignore
        # else:
        # model.test_metrics.update(**outputs)  # type: ignore
        append_to_metrics_(outputs, metrics)

        if batch_idx % print_interval == 0:
            print(
                f"Step: {train_state['global_step']} ({print_str}) Batch: {batch_idx} / {num_iterations}"
            )

    t1 = time.time()
    s_it = (t1 - t0) / num_iterations

    # if self.is_distributed:
    #     metrics = model.module.test_metrics.compute()  # type: ignore
    #     model.module.test_metrics.reset()  # type: ignore
    # else:
    # metrics = model.test_metrics.compute()  # type: ignore
    # model.test_metrics.reset()  # type: ignore
    metrics = apply_metric_fns(metrics, metric_fns, is_training=False)
    metrics[f"s_it"] = s_it

    metrics = printing.add_prefix(metrics, prefix)

    if log_metrics_fn is not None:
        log_metrics_fn(metrics, step=train_state["global_step"])

    return metrics


def train_step(
    train_state, model, optimizer, forward_and_loss_fn, batch, print_interval=32
):
    model.train()
    batch = to_device(batch, train_state["device"])

    loss, result = forward_and_loss_fn(batch, model=model)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if torch.isnan(loss) or torch.isinf(loss):
        train_state["should_raise"] = ValueError("Loss is NaN.")

    # if self.is_distributed:
    #     model.module.train_metrics.update(**outputs)  # type: ignore
    # else:
    append_to_metrics_(result, train_state["train_metrics"])

    if train_state["global_step"] % print_interval == 0:
        print(f"Step: {train_state['global_step']} (Training) Loss: {loss:.4f}")


def should_stop(state, max_steps=None, max_time=None):
    if (
        max_time is not None
        and max_time < datetime.datetime.now() - state["starting_time"]
    ):
        print("Stopping due to max_time.")
        return True
    if max_steps is not None and state["global_step"] >= max_steps:
        print(f"Stopping due to max_steps ({max_steps})")
        return True
    return False


def fit(
    run_dir,
    model,
    optimizer,
    datasets,
    forward_and_loss_fn,
    metrics_fns,
    log_metrics_fn,
    scheduler=None,
    print_interval=32,
    val_check_interval=1024,
    skip_initial_eval=False,
    limit_val_batches=float("inf"),
    max_steps=1,
    max_time=None,
):
    if max_time is not None:
        raise NotImplementedError("max_time is not implemented yet.")
    device = next(model.parameters()).device

    if torch.cuda.is_available() and not device.type == "cuda":
        print("CUDA is available but not being used.")

    train_loader = datasets["train_loader"]
    # if not isinstance(
    #     train_loader.sampler, data.InfiniteRandomSampler
    # ):  # pragma: no cover
    #     print(
    #         "\nWARNING: Training does not use InfiniteRandomSampler, deterministic training disabled."
    #     )

    print("\nModel Summary\n---")
    print(model)
    print(f"Total parameters: {printing.human_format_float(count_parameters(model))}\n")

    t0 = time.time()

    train_state = {
        "global_step": 0,
        "last_global_step": 0,
        "should_raise": None,
        "current_epoch": 0,
        "device": device,
        "train_metrics": defaultdict(list),
    }

    checkpoint_dir = os.path.join(run_dir, "files", "checkpoints")
    # if os.path.exists(checkpoint_dir):
    #     print("Loading previous checkpoint")
    #     checkpoint.load_checkpoint(
    #         checkpoint_dir,
    #         model,
    #         train_state,
    #         optimizer,
    #     )

    keep_training = not should_stop(train_state, max_steps)

    while keep_training:
        # if self.is_distributed:
        #     train_loader.sampler.set_epoch(self.current_epoch)
        # for batch in train_loader:
        # for batch_idx in range(max_steps)
        batch = train_loader[train_state["global_step"]]

        # print(batch[0][0].sum())
        # print(batch[0][-1].sum())
        # if train_state['global_step'] == 8:
        # breakpoint()
        train_step(
            train_state, model, optimizer, forward_and_loss_fn, batch, print_interval
        )

        if scheduler is not None:
            raise NotImplementedError
            scheduler.step()  # pragma: no cover

        lr = optimizer.param_groups[0]["lr"]

        if train_state["global_step"] % print_interval == 0:
            t1 = time.time()
            # if self.is_distributed:
            #     train_metrics = model.module.train_metrics.compute()
            #     model.module.train_metrics.reset()
            # else:
            train_metrics = apply_metric_fns(
                train_state["train_metrics"],
                metrics_fns,
                is_training=True
            )
            s_it = (t1 - t0) / (
                train_state["global_step"] + 1 - train_state["last_global_step"]
            )
            train_metrics["s_it"] = s_it
            train_metrics["lr"] = lr
            train_metrics["epoch"] = train_state["current_epoch"]

            if log_metrics_fn is not None:
                train_metrics = printing.add_prefix(train_metrics, "train")
                log_metrics_fn(train_metrics, step=train_state["global_step"])
            train_state["train_metrics"].clear()

            t0 = time.time()
            train_state["last_global_step"] = train_state["global_step"]

        should_validate = train_state["global_step"] % val_check_interval == 0 and (
            train_state["global_step"] > 0 if skip_initial_eval else True
        )


        if should_validate:
            val_metrics = None
            if datasets["val_loader"] is not None and limit_val_batches > 0:
                if train_state["global_step"] == 0 and skip_initial_eval:
                    print("Skipping initial evaluation.")  # pragma: no cover

                val_metrics = test_loop(
                    train_state,
                    model,
                    forward_and_loss_fn,
                    datasets["val_loader"],
                    metrics_fns,
                    log_metrics_fn,
                    validation=True,
                    limit_batches=limit_val_batches,
                )

            t0 = time.time()
            train_state["last_global_step"] = train_state["global_step"]

            if datasets["test_loader"] is not None and limit_val_batches > 0:
                test_loop(
                    train_state,
                    model,
                    forward_and_loss_fn,
                    datasets["test_loader"],
                    metrics_fns,
                    log_metrics_fn,
                    limit_batches=limit_val_batches,
                )

            checkpoint.checkpoint(
                checkpoint_dir,
                model,
                train_state,
                optimizer,
                metrics=val_metrics,
            )

        train_state["global_step"] += 1
        train_state["batch_index"] = (
            train_state["global_step"]
            * train_loader.batch_size
            // len(train_loader.dataset)
        )

        if train_state["should_raise"] is not None:
            raise train_state["should_raise"]  # pragma: no cover

        if should_stop(train_state, max_steps):
            val_metrics = None
            if datasets["val_loader"] is not None and limit_val_batches > 0:
                val_metrics = test_loop(
                    train_state,
                    model,
                    forward_and_loss_fn,
                    datasets["val_loader"],
                    metrics_fns,
                    log_metrics_fn,
                    validation=True,
                    limit_batches=limit_val_batches,
                )
            # breakpoint()
            checkpoint.checkpoint(
                checkpoint_dir,
                model,
                train_state,
                optimizer,
                metrics=val_metrics,
            )
            keep_training = False
            break

    for k, v in datasets.items():
        if v is not None:
            v.close()

    return True
