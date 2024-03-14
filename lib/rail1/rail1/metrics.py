import torch


def mean_key(metric_dicts, is_training, key):
    # This really should be torch.cat. If it's not, then the metric was already reduced.
    if not metric_dicts[key]:
        return {}
    return {key: torch.mean(torch.cat(metric_dicts[key]))}


def figure_key(metric_dicts, is_training, key):
    return {key: metric_dicts.get(key, {}).pop()} if key in metric_dicts else {}


def inv_accuracy(metric_dicts, is_training):
    if not metric_dicts["logits"] and not metric_dicts["targets"]:
        return {}
    predictions = torch.cat(metric_dicts["logits"])
    targets = torch.cat(metric_dicts["targets"])
    return {
        "inv_accuracy": 1 / (1e-9 + (predictions.argmax(dim=-1) == targets).float().mean())
    }
