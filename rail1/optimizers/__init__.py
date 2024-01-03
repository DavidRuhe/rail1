from torch import optim

def adam(model):
    return optim.Adam(model.parameters())