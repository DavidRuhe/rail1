def pretty_dict(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + k)
            pretty_dict(v, indent + 1)
        else:
            print("  " * indent + f"{k}: {v}")


def human_format_float(num: float):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def add_prefix(d, prefix):
    return {f"{prefix}/{k}": v for k, v in d.items()}


def format_timedelta(td, format_str):
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    formatted_str = format_str.replace("{d}", str(days))
    formatted_str = formatted_str.replace("{h}", f"{hours:02d}")
    formatted_str = formatted_str.replace("{m}", f"{minutes:02d}")
    formatted_str = formatted_str.replace("{s}", f"{seconds:02d}")

    return formatted_str
