config = {
    "name": "anomly",
    "project": "eqvit",
    "entity": "druhe",
    "method": "grid",
    "deterministic": True,
    "device": "cuda",
    "command": ["python", "-u", "anomaly.py", "config/anomaly.py", "${args}"],
    "seed": 0,
    "continue": None,
    "dataset": {"name": "anomaly", "batch_size": 64},
    "model": {"name": "anomaly_transformer"},
    "optimizer": {"name": "adamw", "lr": 5e-4},
    # "optimizer": {"name": "radam", "lr": 5e-4},
    "scheduler": {"max_steps": 100000, "warmup_steps": 10000, "decay_steps": 30000},
    "fit": {"max_steps": float("inf")},
    "parameters": {"seed": {"values": [0]}},
}
