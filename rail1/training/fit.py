import time
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import sampler

from rail1.utils import printing
from rail1.callbacks import checkpoint
import datetime


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


def apply_metric_fns(metric_dicts, metric_fns):
    result = {}
    for fn in metric_fns:
        result.update(fn(metric_dicts))
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
    validation=False,
    print_interval=32,
):
    model.eval()

    num_iterations = int(min(len(test_loader), train_state["limit_val_batches"]))
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

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= train_state["limit_val_batches"]:
            break

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
    metrics = apply_metric_fns(metrics, metric_fns)
    metrics[f"s_it"] = s_it

    metrics = printing.add_prefix(metrics, prefix)

    if log_metrics_fn is not None:
        log_metrics_fn(metrics, step=train_state["global_step"])


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


def should_stop(state):
    if (
        state["max_time"] is not None
        and state["max_time"] < datetime.datetime.now() - state["starting_time"]
    ):
        print("Stopping due to max_time.")
        return True
    if state["max_steps"] is not None and state["global_step"] >= state["max_steps"]:
        print("Stopping due to max_steps.")
        return True
    return False


def fit(
    model,
    optimizer,
    data,
    forward_and_loss_fn,
    train_metrics_fns,
    eval_metrics_fns,
    log_metrics_fn,
    run_dir=None,
    scheduler=None,
    log_interval=32,
    val_check_interval=1024,
    skip_initial_eval=False,
    limit_val_batches=float("inf"),
):
    device = next(model.parameters()).device

    if torch.cuda.is_available() and not device.type == "cuda":
        print("CUDA is available but not being used.")

    train_loader = data["train_loader"]
    if not isinstance(train_loader.sampler, sampler.RandomSampler):
        raise ValueError(
            "Training loader has a non-random sampler!"
        )  # pragma: no cover

    print("\nModel Summary\n---")
    print(model)
    print(f"Total parameters: {printing.human_format_float(count_parameters(model))}\n")

    t0 = time.time()

    train_state = {
        "run_dir": run_dir,
        "global_step": 0,
        "last_global_step": 0,
        "should_raise": None,
        "current_epoch": 0,
        "device": device,
        "train_metrics": defaultdict(list),
        "limit_val_batches": float("inf"),
        "max_time": None,
        "max_steps": 1,
    }

    while not should_stop(train_state):
        # if self.is_distributed:
        #     train_loader.sampler.set_epoch(self.current_epoch)
        for batch in train_loader:
            train_step(train_state, model, optimizer, forward_and_loss_fn, batch)

            if scheduler is not None:
                scheduler.step()  # pragma: no cover

            lr = optimizer.param_groups[0]["lr"]

            if train_state["global_step"] % log_interval == 0:
                t1 = time.time()
                # if self.is_distributed:
                #     train_metrics = model.module.train_metrics.compute()
                #     model.module.train_metrics.reset()
                # else:
                train_metrics = apply_metric_fns(
                    train_state["train_metrics"], train_metrics_fns
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
                if data["val_loader"] is not None and limit_val_batches > 0:
                    if train_state["global_step"] == 0 and skip_initial_eval:
                        print("Skipping initial evaluation.")  # pragma: no cover

                    test_loop(
                        train_state,
                        model,
                        forward_and_loss_fn,
                        data["val_loader"],
                        eval_metrics_fns,
                        log_metrics_fn,
                        validation=True,
                    )

                t0 = time.time()
                last_global_step = train_state["global_step"]

                if "test_loader" in data:
                    raise NotImplementedError("test_loader not implemented yet.")
                    test_loop(model, optimizer, test_loader, validation=False)
                    should_test = False

                if run_dir is not None:
                    checkpoint.checkpoint(model, optimizer)

            train_state["global_step"] += 1

            if train_state["should_raise"] is not None:
                raise train_state["should_raise"]  # pragma: no cover

        train_state["current_epoch"] += 1
