#!/usr/bin/env python3
"""compare_paths.py <your paths.csv> <new paths.csv>
Summarises how role A's regenerated paths.csv differs from the one you have."""
import sys
import pandas as pd

mine, new = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
print(f"rows: yours {len(mine)}, new {len(new)}")
print("clock_domain counts, yours:", mine["clock_domain"].value_counts().sort_index().to_dict())
print("clock_domain counts, new:  ", new["clock_domain"].value_counts().sort_index().to_dict())
if "check_type" in new.columns:
    print("check_type counts, new:    ", new["check_type"].value_counts().sort_index().to_dict())
for col in ("slack_ns", "delay_ns"):
    print(f"{col}: yours min {mine[col].min():.3f} max {mine[col].max():.3f} | "
          f"new min {new[col].min():.3f} max {new[col].max():.3f}")
both = set(mine["endpoint"]) & set(new["endpoint"])
print(f"endpoints in both files: {len(both)} of {new['endpoint'].nunique()} (path_id values are not comparable)")
