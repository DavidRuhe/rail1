config = {
    "name": "random_spheres_suprec",
    "project": "cnf",
    "entity": "druhe",
    "method": "grid",
    "deterministic": False,
    "device": "cuda",
    "command": [
        "python",
        "-u",
        "random_spheres_suprec.py",
        "/".join(__file__.split("/")[-2:]),
        "${args}",
    ],
    "seed": 0,
    "continue": None,
    "dataset": dict(
        name="random_spheres", n_points=1024, batch_size=32, radius_rng=(.2, 1.0)
    ),
    # "model": dict(name="pointnetpp_clean"),
    "model": dict(
        name="cnf", 
        input_dim=3, 
        output_dim=1, 
        input_conditioning_dim=1
    ),
    "optimizer": {"name": "adam", "lr": 1e-3},
    # "scheduler": dict(
    #     name="CosineAnnealingLR",
    #     max_steps=100_000,
    #     warmup_steps=1000,
    #     decay_steps=90_000,
    # ),
    "fit": {
        "max_steps": 100_000,
        "print_interval": 32,
        "limit_val_batches": float("inf"),
    },
    "parameters": {"seed": {"values": [0]}},
    # "cluster": dict(
    #     address="snellius",
    #     directory="/home/druhe/rail1/src/cnf/",
    #     slurm="--partition=gpu --time=24:00:00 --gpus-per-node=1",
    # ),
}
