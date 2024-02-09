import functools

from rail1.utils.path import rglob
from rail1.utils.seed import set_seed


def log_all_if_logger(metrics, step, logger):
    if logger is not None:
        logger.log_all(metrics, step)


def get_logging_fn(logger):
    return functools.partial(log_all_if_logger, logger=logger)
