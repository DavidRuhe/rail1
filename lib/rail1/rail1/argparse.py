import sys
import argparse

from rail1.utils import python
from rail1.utils import printing
from rail1.utils import load_module


def create_parser(d, parser=None, prefix=""):
    assert isinstance(d, dict), d
    if parser is None:
        parser = argparse.ArgumentParser()

    for k in d:
        v = d[k]
        if isinstance(v, dict) and prefix == "":  # Allow just one level of nesting.
            create_parser(v, parser.add_argument_group(prefix + k), prefix + f"{k}.")
        elif isinstance(v, dict):
            for nested_k, nested_v in v.items():
                full_key = f"{prefix}{k}.{nested_k}"
                parser.add_argument(
                    f"--{full_key}",
                    default=nested_v,
                    type=type(nested_v) if nested_v is not None else None,  # type: ignore
                )
        else:
            parser.add_argument(
                f"--{prefix + k}",
                default=v,
                type=type(v) if v is not None else None,  # type: ignore
            )
    return parser


def parse_args():
    argv = sys.argv
    config_path = argv[1]
    config = load_module.load_attribute_from_python_file(config_path, "config")
    del config["parameters"]  # Don't need sweep parameters anymore.

    parser = argparse.ArgumentParser()

    parser = create_parser(config, None)
    args = parser.parse_args(argv[2:])

    config = python.unflatten_dict(vars(args))

    print("\nConfiguration\n---")
    printing.pretty_dict(config)

    # name = get_run_name(sys.argv[1:])
    # experiment = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    return config
