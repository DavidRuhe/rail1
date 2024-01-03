config = {
    "name": "cifar10",
    "project": "pnp",
    "entity": "druhe",
    "method": "grid",
    "deterministic": True,
    "device": "cuda",
    "command": ["python", "-u", "main.py"],
    "dataset": {"name": "cifar10"},
    "model": {"name": "basic_cnn"},
    "optimizer": {"name": "adam"},
    "parameters": {"seed": {"values": [42]}},
}
