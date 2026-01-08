""" Script that displays Table 1 present in the manuscript.
IEO matrices computation should be executed first.
"""
import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("output/table_1_res/")

pattern = re.compile(
    r"knapsack_inst(?P<inst>\d+).*?"
    r"_(?P<method>bc|ccg)_"
    r"ns(?P<ns>\d+)_"
    r"tl(?P<tl>\d+)_per\.csv$"
)

records = []

for csv_file in DATA_DIR.glob("*.csv"):
    match = pattern.match(csv_file.name)
    if not match:
        print("NO MATCH:", csv_file.name)
        continue

    meta = match.groupdict()

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()

    row = df.iloc[0]

    records.append({
        "inst": int(meta["inst"]),
        "ns": int(meta["ns"]),
        "tl": int(meta["tl"]),
        "method": meta["method"],
        "n_cuts": row["n_cuts"],
        "ieo_loss": row["ieo_loss"],
        "mip_gap": row["mip_gap"],
    })


df_long = pd.DataFrame(records)

df_wide = (
    df_long
    .pivot_table(
        index=["inst", "ns", "tl"],
        columns="method",
        values=["ieo_loss", "n_cuts", "mip_gap"],
        aggfunc="first"
    )
)

df_wide.columns = [
    f"{metric}_{method}"
    for metric, method in df_wide.columns
]

df_wide = df_wide.reset_index()

final_df = df_wide[
    [
        "inst",
        "ns",
        "tl",
        "n_cuts_ccg",
        "ieo_loss_ccg",
        "mip_gap_ccg",
        "n_cuts_bc",
        "ieo_loss_bc",
        "mip_gap_bc",
    ]
]

print(final_df)
