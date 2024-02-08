import torch


def default_collate(batch):
    return tuple(
        torch.stack(z) if isinstance(z[0], torch.Tensor) else torch.tensor(z)
        for z in zip(*batch)
    )
