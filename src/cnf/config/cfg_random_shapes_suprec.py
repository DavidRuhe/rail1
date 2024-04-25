config = {
    "name": "random_shapes_suprec",
    "project": "cnf",
    "entity": "druhe",
    "method": "grid",
    "deterministic": False,
    "device": "cuda",
    "command": [
        "python",
        "-u",
        "random_shapes_suprec.py",
        "/".join(__file__.split("/")[-2:]),
        "${args}",
    ],
    "seed": 0,
    "continue": None,
    "dataset": dict(
        name="random_shapes", num_points=1024, batch_size=32
    ),
    # "model": dict(name="pointnetpp_clean"),
    "model": dict(
        # name="siren_cnf", 
        name="shapes_fnf",
        input_dim=3,
        # dim_hidden=256,
        # dim_out=1,
        # num_layers=5,
        # w0_initial=30, 
        output_dim=1, 
        # modulator_dims=(1, 256, 256, 256, 265, 256)
        input_conditioning_dim=6
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
        "clip_grad_norm": 1.0,
    },
    "parameters": {"seed": {"values": [0]}},
    # "cluster": dict(
    #     address="snellius",
    #     directory="/home/druhe/rail1/src/cnf/",
    #     slurm="--partition=gpu --time=24:00:00 --gpus-per-node=1",
    # ),
}
