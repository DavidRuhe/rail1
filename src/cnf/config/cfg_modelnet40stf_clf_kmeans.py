config = {
    "name": "modelnet40stf_clf_kmeans",
    "project": "cnf",
    "entity": "druhe",
    "method": "grid",
    "deterministic": False,
    "device": "cuda",
    "command": [
        "python",
        "-u",
        "modelnet40stf_clf.py",
        __file__,
        "${args}",
    ],
    "seed": 0,
    "continue": None,
    "dataset": dict(
        name="modelnet40stf_points_kmeans",
        num_points=1024,
        batch_size=32,
        num_workers=4,
        n_prefetch=2,
    ),
    "model": dict(name="pointnetpp_clean"),
    "optimizer": {"name": "adam", "lr": 1e-3},
    # "scheduler": {"max_steps": 100_000, "warmup_steps": 1000, "decay_steps": 30000},
    "scheduler": None,
    "fit": {"max_steps": 100_000, "print_interval": 32, "limit_val_batches": float('inf')},
    "parameters": {"seed": {"values": [0]}},
    "cluster": dict(
        address="snellius",
        directory="/home/druhe58/rail1/src/cnf/",
        slurm="--partition=gpu --time=120:00:00 --gpus-per-node=1",
    ),
}