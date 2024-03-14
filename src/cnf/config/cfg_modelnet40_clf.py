config = {
    "name": "modelnet40_clf",
    "project": "cnf",
    "entity": "druhe",
    "method": "grid",
    "deterministic": False,
    "device": "cuda",
    "command": [
        "python",
        "-u",
        "modelnet40_clf.py",
        "config/cfg_modelnet40_clf.py",
        "${args}",
    ],
    "seed": 0,
    "continue": None,
    "dataset": dict(name="modelnet40_points", batch_size=32),
    "model": dict(name="vn_dgcnn"),
    "optimizer": {"name": "adam", "lr": 1e-3},
    # "scheduler": {"max_steps": 100_000, "warmup_steps": 1000, "decay_steps": 30000},
    "scheduler": None,
    "fit": {"max_steps": 100_000, "print_interval": 1, "limit_val_batches": 1},
    "parameters": {"seed": {"values": [0]}},
    "cluster": dict(
        address="snellius",
        directory="/home/druhe/rail1/src/cnf/",
        slurm="--partition=gpu --time=120:00:00 --gpus-per-node=1",
    ),
}