import os, fnmatch


def rglob(directory, pattern, skip_patterns=()):
    assert isinstance(skip_patterns, (list, tuple))
    for root, _, files in os.walk(directory):
        if any(skip_pattern in root for skip_pattern in skip_patterns):
            continue
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def split_path(file, k):
    f = file
    for _ in range(k):
        f = os.path.split(f)[0]
    return f
