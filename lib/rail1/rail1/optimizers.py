from torch import optim
from itertools import chain


def _concat_if_iterable(model):
    if isinstance(model, (list, tuple)):
        return list(chain.from_iterable(m.parameters() for m in model))

    else:
        return model.parameters()


def adam(model, **kwargs):
    return optim.Adam(_concat_if_iterable(model), **kwargs)


def radam(model, **kwargs):
    return optim.RAdam(_concat_if_iterable(model), **kwargs)


def adamw(model, **kwargs):
    return optim.AdamW(_concat_if_iterable(model), **kwargs)


def sgd(model, **kwargs):
    return optim.SGD(_concat_if_iterable(model), **kwargs)
