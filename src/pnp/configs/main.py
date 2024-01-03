config = {
    "name": "cifar10",
    "project": "pnp",
    "entity": "druhe",
    "method": "grid",
    "deterministic": True,
    "device": "cuda",
    "command": ["python", "-u", "main.py", "configs/main.py", "${args}"],
    "dataset": {"name": "cifar10"},
    "model": {"name": "basic_cnn"},
    "optimizer": {"name": "adam"},
    "parameters": {"seed": {"values": [0, 1, 2, 3, 4]}},
}
