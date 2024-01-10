from .transformer import Transformer


def reverse_transformer():
    return Transformer(
        input_dim=10, model_dim=32, num_heads=1, num_classes=10, num_layers=1, dropout=0
    )
