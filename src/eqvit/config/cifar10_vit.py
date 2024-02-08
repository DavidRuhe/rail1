config = {
    "name": "cifar10_vit",
    "project": "eqvit",
    "entity": "druhe",
    "method": "grid",
    "deterministic": True,
    "device": "cuda",
    "command": ["python", "-u", "cifar10_vit.py", "config/cifar10_vit.py", "${args}"],
    "seed": 0,
    "continue": None,
    "dataset": {"name": "cifar10", "batch_size": 128},
    "model": {"name": "cifar10_vit"},
    "optimizer": {"name": "adam", "lr": 5e-4},
    # "optimizer": {"name": "radam", "lr": 5e-4},
    "scheduler": {"max_steps": 100000, "warmup_steps": 0, "decay_steps": 30000},
    "fit": {"max_steps": float("inf")},
    "parameters": {"seed": {"values": [0]}},
}
