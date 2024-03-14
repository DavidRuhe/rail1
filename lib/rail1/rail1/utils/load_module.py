import importlib
from typing import Callable, Any
import sys
import importlib.util


def load_attribute_from_python_file(path, attribute):
    spec = importlib.util.spec_from_file_location(attribute, path)
    assert spec is not None, f"Could not find {attribute} in {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{attribute}_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    attribute = getattr(module, attribute)
    return attribute
