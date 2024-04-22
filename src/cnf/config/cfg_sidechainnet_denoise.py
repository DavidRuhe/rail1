config = {
    "name": "sidechainnet_denoise",
    "project": "cnf",
    "entity": "druhe",
    "method": "grid",
    "deterministic": False,
    "dtype": "float64",
    "device": "cuda",
    "command": [
        "python",
        "-u",
        "sidechainnet_denoise.py",
        '/'.join(__file__.split('/')[-2:]),
        "${args}",
    ],
    "seed": 0,
    "continue": None,
    "dataset": dict(
        name="sidechainnet",
    ),
    "model": dict(name="vn_trafo"),
    "optimizer": {"name": "adam", "lr": 1e-4},
    "scheduler": dict(
        name="CosineAnnealingLR",
        max_steps=100_000,
        warmup_steps=1000,
        decay_steps=90_000,
    ),
    "fit": {"max_steps": 100_000, "print_interval": 32, "limit_val_batches": float('inf')},
    "parameters": {"seed": {"values": [0]}},
    "cluster": dict(
        address="snellius",
        directory="/home/druhe/rail1/src/cnf/",
        slurm="--partition=gpu --time=24:00:00 --gpus-per-node=1",
    ),
}