""" Script that displays Tables 2 and 3 present in the manuscript.
IEO matrices computation, and out-of-sample experimentation
should be executed first.
"""
import pandas as pd
import os

stats_file_path = os.path.join("output", "table_23_res", "final_stats.csv")
df = pd.read_csv(stats_file_path)

# Columns to be displayed
cols = [
    "N. Samples",
    "N. Instance",
    "Delta",
    "Mean in-sample loss (bLR)",
    "Mean in-sample loss (bSPOp)",
    "Mean in-sample loss (bSPOpCH)",
    "Mean in-sample loss (bIEO)",
    "Mean out-of-sample loss (bLR)",
    "Mean out-of-sample loss (bSPOp)",
    "Mean out-of-sample loss (bSPOpCH)",
    "Mean out-of-sample loss (bIEO)",
]

# Table 1: instances without lhs
df_no_lhs = df[~df["N. Instance"].astype(str).str.contains("_lhs", regex=False)]
table_no_lhs = df_no_lhs[cols]

# Table 2: instances with lhs
df_lhs = df[df["N. Instance"].astype(str).str.contains("_lhs", regex=False)]
table_lhs = df_lhs[cols]

# Display tables

print("\n=== Instances without LHS ===")
print(table_no_lhs.to_string(index=False))

print("\n=== Instances with LHS ===")
print(table_lhs.to_string(index=False))