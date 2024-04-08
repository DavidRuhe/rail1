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
        name="modelnet40stf_points",
        resolutions=[1024, 512, 256, 128],
        batch_size=32,
        num_workers=32,
        n_prefetch=4,
    ),
    # "model": dict(name="pointnetpp_clean"),
    "model": dict(name="pointmlp"),
    "optimizer": {"name": "adam", "lr": 1e-3},
    # "scheduler": {"max_steps": 100_000, "warmup_steps": 1000, "decay_steps": 30000},
    "scheduler": None,
    "fit": {"max_steps": 100_000, "print_interval": 32, "limit_val_batches": float('inf')},
    "parameters": {"seed": {"values": [0]}},
    # "cluster": dict(
    #     address="snellius",
    #     directory="/home/druhe58/rail1/src/cnf/",
    #     slurm="--partition=gpu --time=00:50:00 --gpus-per-node=1",
    # ),
}
