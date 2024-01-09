def unflatten(dictionary, sep="."):
    result = dict()
    for k, v in dictionary.items():
        parts = k.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = v
    return result