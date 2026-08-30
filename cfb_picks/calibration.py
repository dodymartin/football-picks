import json

EDGE_BUCKETS = [
    (2, "0-2"),
    (5, "2-5"),
    (8, "5-8"),
    (15, "8-15"),
    (25, "15-25"),
    (float("inf"), "25+"),
]


def edge_bucket(abs_edge):
    for upper, name in EDGE_BUCKETS:
        if abs_edge < upper:
            return name
    return EDGE_BUCKETS[-1][1]


def save_calibration(calibration, path):
    with open(path, "w") as f:
        json.dump(calibration, f)


def load_calibration(path):
    with open(path) as f:
        return json.load(f)
