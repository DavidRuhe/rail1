import itertools
import subprocess
import sys

from rail1.utils import load_module, versioning

from .run import process_args_and_load_config


def main():  # pragma: no cover
    versioning.check_git_detached()
    config, name, project, entity = process_args_and_load_config(sys.argv, devrun=True)

    parameters = config["parameters"]
    base_command = config["command"]
    # for i, c in enumerate(base_command):
    #     if c == "${env}":
    #         base_command[i] = "/usr/bin/env"
    #     elif c == "${interpreter}":
    #         base_command[i] = "python -u"
    #     elif c == "${program}":
    #         base_command[i] = config["program"]
    #     elif c == "${args}":
    #         del base_command[i]

    args = sys.argv[2:]

    for k, v in parameters.items():
        parameters[k] = parameters[k]["values"]

    if parameters:
        keys, values = zip(*parameters.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for d in permutations_dicts:
            print("\nRunning with configuration:")
            print(d)
            print()
            command = base_command + [f"--{k}={v}" for k, v in d.items()]
            command = " ".join(command + args)
            result = subprocess.call(command, shell=True)

            if result != 0:
                break


if __name__ == "__main__":  # pragma: no cover
    main()
