import torch


def mean_key(metric_dicts, is_training, key):
    # This really should be torch.cat. If it's not, then the metric was already reduced.
    if not metric_dicts[key]:
        return {}
    return {key: torch.mean(torch.cat(metric_dicts[key]))}


def figure_key(metric_dicts, is_training, key):
    return {key: metric_dicts.get(key, {}).pop()} if key in metric_dicts else {}


def accuracy(metric_dicts, is_training):
    """
    Calculates the accuracy metric for a given set of predictions and targets.

    Args:
        metric_dicts (dict): A dictionary containing the keys "logits" and "targets", which are lists of tensors.
        is_training (bool): A flag indicating whether the model is in training mode.

    Returns:
        dict: A dictionary containing the accuracy metric.

    Raises:
        AssertionError: If the dimensions of the targets and predictions are not as expected.
    """
    if not metric_dicts["logits"] and not metric_dicts["targets"]:
        return {}
    predictions = torch.cat(metric_dicts["logits"])
    targets = torch.cat(metric_dicts["targets"])
    assert targets.dim() == 1
    assert predictions.dim() == 2
    assert len(targets) == len(predictions)
    return {"accuracy": (predictions.argmax(dim=-1) == targets).float().mean()}


def binary_accuracy(metric_dicts, is_training):
    """
    Calculates the binary accuracy metric for a given set of predictions and targets.

    Args:
        metric_dicts (dict): A dictionary containing the keys "logits" and "targets", which are lists of tensors.
        is_training (bool): A flag indicating whether the model is in training mode.

    Returns:
        dict: A dictionary containing the binary accuracy metric.

    Raises:
        AssertionError: If the dimensions of the targets and predictions are not as expected.
    """
    if not is_training:
        assert metric_dicts["logits"] and metric_dicts["targets"]
    if not metric_dicts["logits"] and not metric_dicts["targets"]:
        return {}
    predictions = torch.cat(metric_dicts["logits"])
    targets = torch.cat(metric_dicts["targets"])
    assert targets.shape == predictions.shape
    result =  {"binary_accuracy": (predictions > 0).float().eq(targets).float().mean()}
    return result
