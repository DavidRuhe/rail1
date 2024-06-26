import warnings
import itertools
import subprocess
import os
import re
import sys

import yaml

import wandb
from rail1.utils import versioning, load_module


# def get_current_git_branch():
#     try:
#         output = subprocess.check_output(["git", "branch", "--show-current"])
#         current_branch = output.strip().decode("utf-8")
#     except subprocess.CalledProcessError:
#         current_branch = None
#     return current_branch


def generate_sbatch_lines(slurm_args_str):
    slurm_args = slurm_args_str.split()
    sbatch_lines = ["#!/bin/bash"]

    i = 0
    while i < len(slurm_args):
        arg = slurm_args[i]
        if arg.startswith("--"):
            arg_name = arg[2:]
            if "=" in arg_name:
                arg_name, arg_value = arg_name.split("=")
                sbatch_lines.append(f"#SBATCH --{arg_name}={arg_value}")
            else:
                arg_value = slurm_args[i + 1]
                sbatch_lines.append(f"#SBATCH --{arg_name}={arg_value}")
                i += 1
        elif arg.startswith("-"):
            arg_name = arg[1:]
            arg_value = slurm_args[i + 1]
            sbatch_lines.append(f"#SBATCH -{arg_name} {arg_value}")
            i += 1
        i += 1

    return sbatch_lines


# def push_files(sweep_id):
#     command = f"""
#         find * -size -4M -type f -print0 | xargs -0 git add
#         git add -u
#         git commit --allow-empty -m {sweep_id}
#         git push
#     """
#     os.system(command)


def commit_files(sweep_id):  # pragma: no cover
    command = f"""
        find * -size -4M -type f -print0 | xargs -0 git add
        git add -u
        git commit --allow-empty -m {sweep_id}
        git tag {sweep_id}
        git push
        git push origin {sweep_id}
    """
    subprocess.check_call(command, shell=True)
    # # Get the latest commit SHA
    # commit_sha = subprocess.getoutput("git rev-parse HEAD")
    # return commit_sha


def write_jobfile(slurm_string, n_jobs, command, directory, sweep_id):
    sbatch_lines = generate_sbatch_lines(slurm_string)

    sbatch_lines.append(f"#SBATCH --array=1-{n_jobs}")
    sbatch_lines.append(f"#SBATCH --output={os.path.join(directory, 'slurm-%j.out')}")
    sbatch_lines.append(f"cd {directory}")
    sbatch_lines.append(f"git checkout {sweep_id}")
    sbatch_lines.append("source ./activate.sh")
    sbatch_lines.append(str(command))

    sbatch_script = "\n".join(sbatch_lines)

    jobfile_path = os.path.join(directory, "slurm_job.sh")
    with open(jobfile_path, "w") as f:
        f.write(sbatch_script)


def replace_variables(command, locals):
    # Use a regular expression to find words between "{}" brackets in the string
    pattern = re.compile(r"\{([^}]+)\}")
    matches = pattern.findall(command)

    # For each word found, replace it with its value from the locals_dict dictionary
    for match in matches:
        command = command.replace("{" + match + "}", str(locals[match]))
    return command


def git_status():
    # Fetch the remote changes without applying them
    subprocess.run(
        ["git", "fetch"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Get the status comparing local and remote
    status = subprocess.getoutput("git rev-list --left-right --count HEAD...@{u}")

    local, remote = map(int, status.split("\t"))
    if local and remote:
        return "diverged"
    elif local:
        return "ahead"
    elif remote:
        return "behind"
    else:
        return "up-to-date"


def process_args_and_load_config(argv, devrun=False):
    versioning.check_git_detached()

    if not devrun:
        status = git_status()
        if status == "behind":
            warning = f"git is behind remote. Please pull changes first."
        elif status == "diverged":
            warning = f"git has diverged from remote. Please push or pull changes."
        else:
            warning = None

        if warning is not None:
            warnings.warn(warning)
            cont = input("Continue? [y/N]")
            if cont.lower() != "y":
                raise RuntimeError("Aborting.")

    if not devrun and len(argv) != 2 and "-h" not in argv:
        raise ValueError(
            f"Usage: sweep <config.yaml>. Please don't provide any other arguments."
        )
    config_path = argv[1]
    config = load_module.load_attribute_from_python_file(config_path, "config")

    name = config["name"]
    project = config["project"]
    entity = config["entity"]

    return config, name, project, entity


def main():
    config, name, project, entity = process_args_and_load_config(sys.argv)

    sweep_id = wandb.sweep(sweep=config, project=project, entity=entity)
    on_cluster = "cluster" in config

    if on_cluster:
        raise NotImplementedError
        cluster_config = config["cluster"]
        command = cluster_config["command"]
        slurm_arguments = cluster_config["slurm"]
        directory = cluster_config["directory"]
        all_values = [config["parameters"][k]["values"] for k in config["parameters"]]
        num_jobs = len(tuple(itertools.product(*all_values)))
        command = replace_variables(command, locals())
        write_jobfile(slurm_arguments, num_jobs, command, directory, sweep_id)
    else:
        command = "WANDB_ENABLED=TRUE wandb agent {entity}/{project}/{sweep_id}"
        command = replace_variables(command, locals())
        cluster_config = None
        directory = None

    commit_files(sweep_id)

    if on_cluster:
        raise NotImplementedError
        assert cluster_config is not None
        assert directory is not None
        print("\nSuccessfully submitted sweep. To fire remotely, run:")
        print(f"ssh {cluster_config['address']}")
        print(
            f"cd {directory} && git fetch && git checkout {sweep_id} && sbatch slurm_job.sh\n"
        )

    else:
        print(f"Run this sweep with:")
        print(f"git checkout {sweep_id} && {command}")


if __name__ == "__main__":  # pragma: no cover
    main()
