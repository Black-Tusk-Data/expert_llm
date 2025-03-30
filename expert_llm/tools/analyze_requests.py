import json


def load_reqs_file(fname: str):
    import pandas as pd

    lines: list[str] = []
    with open(fname, "r") as f:
        lines = f.read().split("\n")
        pass
    rows: list[dict] = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except Exception as e:
            print("failed to parse line", line, e)
            pass
        pass
    df = pd.DataFrame.from_records(rows)
    return df
