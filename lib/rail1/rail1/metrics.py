import torch


def mean_key(metric_dicts, is_training, key):
    return {key: torch.mean(torch.cat(metric_dicts[key]))}


def mean_loss(metric_dicts, is_training):
    return mean_key(metric_dicts, "loss")


def figure_key(metric_dicts, is_training, key):
    return {key: metric_dicts.get(key, {}).pop()} if key in metric_dicts else {}
