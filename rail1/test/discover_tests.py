import ast
import sys
from rail1.utils import rglob, load_module
import os
import unittest


def extract_def_class_lines(code):
    tree = ast.parse(code)
    lines = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, "end_lineno") else start_line
            assert end_line is not None
            lines.extend(range(start_line, end_line + 1))

    return lines


def filter_lines(lines):
    code = "".join(lines)
    valid_lines = extract_def_class_lines(code)

    def is_valid(line_number):
        line = lines[line_number - 1]
        line = line.strip()
        if not line:
            return False
        if line.endswith("# pragma: no cover"):
            return False
        if line == ")":
            return False
        if line == "}":
            return False
        if line == "else:":
            return False
        if line.startswith("#"):
            return False
        return True

    valid_lines = [line_number for line_number in valid_lines if is_valid(line_number)]

    return valid_lines


def main():
    if len(sys.argv) != 2:
        raise ValueError("Please provide path to run files.")
    path = sys.argv[1]
    if os.path.isdir(path):
        files = tuple(rglob(path, "*.py", skip_patterns=(".venv", "tests")))
    else:
        files = [path]

    print(f"Running tests in {len(files)} files.")
    files = [os.path.abspath(f) for f in files]
    executed_lines = {}

    def trace_lines(frame, event, arg):
        if event == "line":
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            if filename not in executed_lines:
                executed_lines[filename] = set()
            executed_lines[filename].add(lineno)
        return trace_lines

    def run_tests(path):
        sys.settrace(trace_lines)  # Set the trace function
        # for f in files:
        #     path, module = os.path.split(f)
        #     sys.path.append(path)
        #     module_name = os.path.splitext(module)[0]
        #     module = __import__(module_name)
        #     if hasattr(module, "test"):
        #         module.test()
        #     sys.path.remove(path)
        #     del sys.modules[module_name]
        loader = unittest.TestLoader()
        suite = loader.discover(path)
        runner = unittest.TextTestRunner()
        runner.run(suite)

        sys.settrace(None)  # Disable the trace function

    run_tests(path)

    executed_lines = {k: v for k, v in executed_lines.items() if k in files}
    lines_not_covered = {}

    def calculate_coverage(files):
        total_lines = 0
        covered_lines = 0

        for f in files:
            with open(f, "r") as file:
                lines = file.readlines()

                lines = filter_lines(lines)
                if f in executed_lines:
                    executed_lines_f = executed_lines[f]
                else:
                    executed_lines_f = set()

                is_covered = [line_number in executed_lines_f for line_number in lines]

                covered_lines += sum(is_covered)
                total_lines += len(lines)

                not_covered = [line for line, is_covered in zip(lines, is_covered) if not is_covered]

                if not_covered:
                    lines_not_covered[f] = not_covered

        coverage = covered_lines / total_lines * 100
        print(lines_not_covered)
        return coverage

    coverage = calculate_coverage(files)
    print(coverage)

    print("All tests completed!")


if __name__ == "__main__":
    main()
