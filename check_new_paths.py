"""
Acceptance gate for a re-exported paths.csv from Role A.

Run this BEFORE regenerating anything:

    python check_new_paths.py                 # checks ./paths.csv
    python check_new_paths.py paths_new.csv   # checks a file before overwriting

Exits 0 if the file is fit to run with ALLOW_UNVALIDATED_PATHS = False.
Exits 1 otherwise, naming exactly what Role A still needs to fix.

Also forecasts whether droop will actually reorder the ranking, which is the
thing the whole PRISM thesis rests on.
"""
import sys
import os
import numpy as np
import pandas as pd

PATHS = sys.argv[1] if len(sys.argv) > 1 else "paths.csv"
STATS = "design_stats.csv"
INSTS = "instances.csv"

REQUIRED = ["path_id", "endpoint", "clock_domain", "slack_ns", "delay_ns", "inst_ids"]

fail = []
warn = []


def check(ok, msg):
    if ok:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        fail.append(msg)


def note(msg):
    print(f"  WARN  {msg}")
    warn.append(msg)


print(f"\n=== Acceptance gate for {PATHS} ===\n")

if not os.path.exists(PATHS):
    sys.exit(f"FATAL: {PATHS} not found")

df = pd.read_csv(PATHS)
st = pd.read_csv(STATS)
period = float(st["clock_period_ns"].iloc[0])
vdd = float(st["vdd_v"].iloc[0])

print(f"[schema]  {len(df)} rows, clock_period_ns={period}, vdd_v={vdd}")
missing = [c for c in REQUIRED if c not in df.columns]
check(not missing, f"all required columns present (missing: {missing or 'none'})")
if missing:
    sys.exit(1)

# --- the two defects that triggered the banner -------------------------------
print("\n[the two blocking defects]")
n_neg = int((df["delay_ns"] < 0).sum())
check(n_neg == 0,
      f"delay_ns is non-negative on every row (negative on {n_neg}/{len(df)}, "
      f"min={df['delay_ns'].min():.2f})")

# Load per-clock-domain periods if clock_periods.csv exists
target_dir = os.path.dirname(PATHS) or "."
clk_file = os.path.join(target_dir, "clock_periods.csv")
if not os.path.exists(clk_file) and os.path.exists("clock_periods.csv"):
    clk_file = "clock_periods.csv"

if os.path.exists(clk_file):
    df_clk = pd.read_csv(clk_file)
    clk_map = dict(zip(df_clk["clock_domain"], df_clk["period_ns"].astype(float)))
    max_period = df_clk["period_ns"].max()
    print(f"[multi-clock] Loaded domain periods from {clk_file}: {clk_map}")

    def get_path_period(row):
        ep = str(row["endpoint"])
        cd = str(row["clock_domain"])
        if "u_host_if" in ep or "host" in ep:
            return clk_map.get("clk_host", 10.0)
        if "u_nand_if" in ep or "nand" in ep:
            return clk_map.get("clk_nand", 20.0)
        return clk_map.get(cd, max_period)

    path_periods = df.apply(get_path_period, axis=1)
    over_mask = df["slack_ns"] > path_periods
    n_over = int(over_mask.sum())
    max_over_val = df.loc[over_mask, "slack_ns"].max() if n_over > 0 else 0.0
    check(n_over == 0,
          f"slack_ns <= clock_domain period_ns on every row "
          f"({n_over} rows exceed domain period, max={max_over_val:.2f})")
else:
    n_over = int((df["slack_ns"] > period).sum())
    check(n_over == 0,
          f"slack_ns <= clock_period_ns on every row "
          f"({n_over} rows exceed {period} ns, max={df['slack_ns'].max():.2f})")

# --- broader sanity ----------------------------------------------------------
print("\n[sanity]")
check(df["slack_ns"].notna().all(), "no NaN in slack_ns")
check(df["delay_ns"].notna().all(), "no NaN in delay_ns")
check(df["path_id"].is_unique, "path_id is unique")

if os.path.exists(INSTS):
    inst_ids = set(pd.read_csv(INSTS, usecols=["inst_id"])["inst_id"].astype(int))
    used = set()
    for v in df["inst_ids"].astype(str):
        used.update(int(x) for x in v.replace(";", " ").replace(",", " ").split()
                    if x.strip().isdigit())
    orphans = used - inst_ids
    check(not orphans,
          f"every inst_id in paths.csv exists in instances.csv "
          f"({len(orphans)} orphans)")
else:
    note("instances.csv not found - skipped inst_id cross-check")

if (df["slack_ns"] < 0).any():
    note(f"{int((df['slack_ns'] < 0).sum())} paths have NEGATIVE slack - the design "
         f"fails timing before droop is even considered. That is a real result, not "
         f"an error, but say so explicitly on the dashboard.")

domains = df["clock_domain"].nunique()
if domains == 1:
    note(f"only one clock domain ({df['clock_domain'].iloc[0]}) - CDC paths may be missing")

# --- will droop actually matter now? ----------------------------------------
print("\n[forecast: will droop-aware ranking differ from plain slack ranking?]")
print("  Using the current engine's measured penalty range (3.7 ps mean, 13.1 ps max).")
MEAN_PEN_NS, MAX_PEN_NS = 0.003665, 0.013137

s = df["slack_ns"].to_numpy(float)
gaps = np.diff(np.sort(s))
tight = int((gaps < MAX_PEN_NS).sum())
n_unique = df["slack_ns"].nunique()
n_unique_paths = df["inst_ids"].nunique()

print(f"  adjacent-slack gaps smaller than max penalty : {tight}/{len(gaps)}  "
      f"(UPPER BOUND ONLY - see below)")
print(f"  distinct slack values                        : {n_unique}/{len(df)}")
print(f"  distinct inst_ids lists                      : {n_unique_paths}/{len(df)}")
print(f"  max penalty as % of tightest slack           : "
      f"{MAX_PEN_NS / max(abs(s).min(), 1e-9) * 100:.2f}%")

# Ties defeat the gap heuristic: paths sharing a slack value AND an inst_ids list
# get identical penalties, so they cannot reorder relative to each other. On the
# pre-fix file this heuristic predicted ~1568 reorderable pairs and the engine
# measured ZERO. Treat the number above as a ceiling, never as an estimate.
tie_heavy = n_unique < len(df) * 0.25 or n_unique_paths < len(df) * 0.9
if tight == 0:
    note("droop is far too small to reorder ANY path. The droop-aware ranking will "
         "stay identical to plain STA slack ranking. Check with Role B whether "
         "predictions.csv covers this design.")
elif tie_heavy:
    note(f"heavy ties ({n_unique} distinct slack values, {n_unique_paths} distinct "
         f"instance lists across {len(df)} rows). Tied paths get identical penalties "
         f"and cannot reorder, so real churn will be FAR below {tight}. "
         f"The only authoritative number is prism_rank_churn.csv after a run.")
else:
    print(f"  --> up to ~{tight} adjacent pairs could reorder. Confirm against "
          f"prism_rank_churn.csv; do not quote this number.")

# --- verdict -----------------------------------------------------------------
print("\n" + "=" * 62)
if fail:
    print(f"REJECTED - {len(fail)} blocking defect(s). Send back to Role A:\n")
    for m in fail:
        print(f"  - {m}")
    print("\nKeep ALLOW_UNVALIDATED_PATHS = True until these are fixed.")
    print("=" * 62)
    sys.exit(1)

print("ACCEPTED - this file passes schema validation.")
if warn:
    print(f"\n{len(warn)} warning(s) worth reading above, but none are blocking.")
print("\nNext steps:")
print("  1. set ALLOW_UNVALIDATED_PATHS = False  in prism_risk_engine.py")
print("  2. python prism_risk_engine.py")
print("  3. python telemetry_sensor_placement.py")
print("  4. python dse_tier3_bo.py")
print("  5. python verify_prism.py")
print("  6. confirm the UNVALIDATED banner is gone from the dashboard")
print("=" * 62)
sys.exit(0)
