import pandas as pd
import numpy as np
import os
import json
from dataclasses import dataclass

# ==========================================
# IO_CSV.PY — DATA VALIDATION & INGESTION
# Conforms to DATA_SCHEMA.md Contract
# ==========================================

@dataclass
class Design:
    design_id: str
    stats: pd.DataFrame
    paths: pd.DataFrame
    predictions: pd.DataFrame = None


def validate_design(design_dir) -> list:
    """
    Validates a design directory against the PRISM DATA_SCHEMA.md contract.
    Returns a list of error strings. An empty list indicates validation passed.
    """
    errors = []
    
    # 1. Check required files
    stats_path = os.path.join(design_dir, 'design_stats.csv')
    paths_path = os.path.join(design_dir, 'paths.csv')

    if not os.path.exists(stats_path):
        errors.append(f"[ERROR] Required file missing: '{stats_path}'")
    else:
        df_stats = pd.read_csv(stats_path)
        required_stats_cols = ['design_id', 'cells', 'chip_area_um2']
        for col in required_stats_cols:
            if col not in df_stats.columns:
                errors.append(f"[ERROR] 'design_stats.csv' missing required column: '{col}'")

    if not os.path.exists(paths_path):
        errors.append(f"[ERROR] Required file missing: '{paths_path}'")
    else:
        df_paths = pd.read_csv(paths_path)
        required_paths_cols = ['endpoint', 'slack_ns', 'delay_ns']
        for col in required_paths_cols:
            if col not in df_paths.columns:
                errors.append(f"[ERROR] 'paths.csv' missing required column: '{col}'")

        # Sanity check: inst_ids or inst_idx must be present
        if 'inst_ids' not in df_paths.columns and 'inst_idx' not in df_paths.columns:
            errors.append("[ERROR] 'paths.csv' must contain 'inst_ids' or 'inst_idx' column for Role C timing attribution.")

    # Check predictions.csv if present
    pred_path = os.path.join(design_dir, 'predictions.csv')
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path, comment='#')
        if 'tile_id' not in df_pred.columns or 'pred_v' not in df_pred.columns:
            errors.append("[ERROR] 'predictions.csv' must contain columns 'tile_id' and 'pred_v'.")
        else:
            # Check unit sanity: pred_v should be in Volts (< 1.0 V)
            if df_pred['pred_v'].max() > 1.0:
                errors.append(f"[UNIT SANITY ERROR] max(pred_v) = {df_pred['pred_v'].max()} > 1.0 V. Role B exported millivolts instead of Volts!")


    return errors


def load_design(design_dir='.') -> Design:
    """
    Loads and validates a design directory.
    Fails loudly with detailed error messages if validation fails.
    """
    errors = validate_design(design_dir)
    if errors:
        print("\n============================================")
        print(" DATA INTEGRITY VALIDATION FAILURES DETECTED ")
        print("============================================")
        for err in errors:
            print(err)
        print("============================================\n")

    stats_path = os.path.join(design_dir, 'design_stats.csv')
    paths_path = os.path.join(design_dir, 'paths.csv')
    pred_path = os.path.join(design_dir, 'predictions.csv')

    df_stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else None
    df_paths = pd.read_csv(paths_path) if os.path.exists(paths_path) else None
    df_pred = pd.read_csv(pred_path, comment='#') if os.path.exists(pred_path) else None


    design_id = df_stats['design_id'].iloc[0] if df_stats is not None and 'design_id' in df_stats.columns else 'default_design'

    return Design(design_id=design_id, stats=df_stats, paths=df_paths, predictions=df_pred)


def load_corpus(data_dir='.') -> list:
    """Loads all valid design folders in a corpus directory."""
    designs = []
    if os.path.exists(os.path.join(data_dir, 'design_stats.csv')):
        designs.append(load_design(data_dir))
    else:
        for entry in os.listdir(data_dir):
            full_path = os.path.join(data_dir, entry)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'design_stats.csv')):
                designs.append(load_design(full_path))
    return designs


if __name__ == "__main__":
    print("============================================")
    print("      PRISM IO_CSV VALIDATION CHECK        ")
    print("============================================\n")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    errs = validate_design(cur_dir)
    if not errs:
        print("[SUCCESS] All files conform strictly to DATA_SCHEMA.md contract!")
        design = load_design(cur_dir)
        print(f"Loaded Design ID : {design.design_id}")
        print(f"Paths Analyzed   : {len(design.paths) if design.paths is not None else 0}")
    else:
        print(f"Found {len(errs)} validation warnings/errors.")
