import torch
import torch.utils.data.dataloader

# def default_collate(batch):
#     return tuple(
#         torch.stack(z) if isinstance(z[0], torch.Tensor) else torch.tensor(z)
#         for z in zip(*batch)
#     )

def default_collate(batch):
    return torch.utils.data.dataloader.default_collate(batch)