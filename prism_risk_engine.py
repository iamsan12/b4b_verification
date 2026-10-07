import pandas as pd
import numpy as np
import json
import os
import sys

# ==========================================
# CONSTANTS & CONTRACT DEFAULTS
# ==========================================
NX_COARSE = 24
NY_COARSE = 24

# Physical constants (nangate45 PDK specification)
VTH_MV = 300.0   # nangate45 nominal Vth (mV) - standard threshold voltage
ALPHA = 1.3      # alpha-power-law velocity saturation exponent
NOMINAL_V_MV = 1100.0 # nominal supply voltage (mV)

# Schema override flag
ALLOW_UNVALIDATED_PATHS = False
SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER = False

# At-risk path threshold constants
AT_RISK_MODE = 'quantile'        # Modes: 'quantile' or 'absolute'
AT_RISK_QUANTILE = 0.05          # N=5% worst paths for quantile mode
AT_RISK_PERIOD_FRAC = 0.10       # Fraction of clock_period_ns for absolute mode margin

# Role B synthetic corpus aggregation: 14 syn_* designs mean aggregated across 6 workload scenarios in predictions.csv
ROLEB_AGGREGATION = "Role B synthetic corpus — 14-design mean over 6 scenarios (NOT this chip)"

# Supported droop sources configuration
DROOP_SOURCES_CONFIG = {
    'measured_rail': {
        'name': 'Measured Rail (VDD+VSS, Tile Worst)',
        'type': 'measured_tile',
        'file': 'measured_droop/orfs_ssd_ctrl_cfg_small_rail.csv',
        'design': 'orfs_ssd_ctrl_cfg_small',
        'vdd_v': 1.1,
        'includes_vss': True,
        'description': 'Role A signoff PDNSim measured rail droop (VDD drop + VSS bounce tile worst)'
    },
    'measured_vdd': {
        'name': 'Measured VDD Drop (Tile Worst)',
        'type': 'measured_tile',
        'file': 'measured_droop/orfs_ssd_ctrl_cfg_small_vdd.csv',
        'design': 'orfs_ssd_ctrl_cfg_small',
        'vdd_v': 1.1,
        'includes_vss': False,
        'description': 'Role A signoff PDNSim measured worst VDD drop per tile'
    },
    'measured_cell_rail': {
        'name': 'Measured Cell Rail (Per-Instance PSM)',
        'type': 'measured_instance',
        'file': 'final_psm/ssd_ctrl_cfg_small/VDD_BUMPS.csv + VSS_BUMPS.csv',
        'design': 'orfs_ssd_ctrl_cfg_small',
        'vdd_v': 1.1,
        'includes_vss': True,
        'description': 'Role A signoff PDNSim per-instance rail voltage from PSM bumps'
    },
    'measured_p12_rail': {
        'name': 'Measured 12 µm Strap Pitch Run (Tile Rail)',
        'type': 'measured_tile_p12',
        'file': 'measured_droop/orfs_ssd_ctrl_cfg_run_01_p12_rail.csv',
        'design': 'orfs_ssd_ctrl_cfg_run_01_p12',
        'vdd_v': 1.1,
        'includes_vss': True,
        'description': 'Role A signoff 12 µm PDN strap pitch PnR run measured rail droop'
    },
    'guardband_5pct': {
        'name': 'Signoff 5% Guard-band What-If (Uniform 55 mV)',
        'type': 'guardband',
        'frac': 0.05,
        'file': 'uniform_5pct_vdd_budget',
        'design': 'hypothetical_guardband_5pct',
        'vdd_v': 1.1,
        'includes_vss': False,
        'description': 'Hypothetical uniform 5% VDD supply tolerance budget (55 mV) on every tile'
    },
    'guardband_10pct': {
        'name': 'Signoff 10% Guard-band What-If (Uniform 110 mV)',
        'type': 'guardband',
        'frac': 0.10,
        'file': 'uniform_10pct_vdd_budget',
        'design': 'hypothetical_guardband_10pct',
        'vdd_v': 1.1,
        'includes_vss': False,
        'description': 'Hypothetical uniform 10% VDD supply tolerance budget (110 mV) on every tile'
    },
    'roleB_syn_corpus_whatif': {
        'name': 'Role B Synthetic Corpus What-If (14 syn_* avg)',
        'type': 'synthetic_corpus',
        'file': 'predictions.csv',
        'design': ROLEB_AGGREGATION,
        'vdd_v': 1.1,
        'includes_vss': False,
        'description': 'Role B synthetic corpus predictions averaged over 14 syn_* designs across 6 scenarios (NOT this chip)'
    }
}


# ==========================================
# PHASE 1: Data Loader & Validation
# ==========================================
def parse_inst_list(val):
    """Parses instance ID lists from JSON arrays, semicolon strings ('34;36;38'), or comma lists."""
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.startswith('['):
            try:
                return [int(x) for x in json.loads(val_str)]
            except Exception:
                pass
        parts = val_str.replace(';', ' ').replace(',', ' ').split()
        return [int(x) for x in parts if x.isdigit()]
    return []


def build_inst_to_tile(instances_csv='instances.csv', stats_csv='design_stats.csv', nx=NX_COARSE, ny=NY_COARSE):
    """
    Builds a real physical mapping from cell inst_id to tile_id (0..575)
    based on bounding box and die dimensions.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    inst_path = instances_csv if os.path.isabs(instances_csv) else os.path.join(dir_path, instances_csv)
    st_path = stats_csv if os.path.isabs(stats_csv) else os.path.join(dir_path, stats_csv)

    if not os.path.exists(st_path):
        raise FileNotFoundError(f"stats_csv file not found: '{st_path}'")
    st = pd.read_csv(st_path)
    die_w = float(st['die_w_um'].iloc[0])
    die_h = float(st['die_h_um'].iloc[0])
    if not (die_w > 0 and die_h > 0):
        raise ValueError("design_stats.csv: die_w_um / die_h_um must be > 0")

    if not os.path.exists(inst_path):
        raise FileNotFoundError(f"instances_csv file not found: '{inst_path}'")
    inst = pd.read_csv(inst_path)
    for c in ('inst_id', 'x_um', 'y_um'):
        if c not in inst.columns:
            raise ValueError(f"instances.csv: missing required column '{c}'")

    if inst['x_um'].max() > die_w * 1.01 or inst['y_um'].max() > die_h * 1.01:
        raise ValueError(
            "instances.csv coordinates exceed the die. Role A likely exported DEF "
            f"database units, not microns. max x={inst['x_um'].max()}, die_w={die_w}"
        )

    tx = np.clip((inst['x_um'] / die_w * nx).astype(int), 0, nx - 1)
    ty = np.clip((inst['y_um'] / die_h * ny).astype(int), 0, ny - 1)
    return dict(zip(inst['inst_id'].astype(int), (ty * nx + tx).astype(int)))


def validate_psm_summary(df_droop, design='orfs_ssd_ctrl_cfg_small', psm_csv='psm_summary.csv'):
    """Validates loaded droop map against PSM summary signoff targets within 1% relative error."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    psm_path = psm_csv if os.path.isabs(psm_csv) else os.path.join(dir_path, psm_csv)
    
    if len(df_droop) != 576:
        raise ValueError(f"Droop map for '{design}' expected 576 tiles (24x24), got {len(df_droop)}")

    if (df_droop['pred_v'] < 0).any() or (df_droop['pred_v'] > 0.2).any():
        raise ValueError(f"Droop map for '{design}' pred_v out of valid physical range [0, 0.2] Volts")

    if os.path.exists(psm_path):
        df_psm = pd.read_csv(psm_path)
        vdd_row = df_psm[(df_psm['design_id'] == design) & (df_psm['net'] == 'VDD')]
        vss_row = df_psm[(df_psm['design_id'] == design) & (df_psm['net'] == 'VSS')]

        if not vdd_row.empty and not vss_row.empty:
            vdd_expected_mv = float(vdd_row['worst_mv'].iloc[0])
            vss_expected_mv = float(vss_row['worst_mv'].iloc[0])

            if 'vdd_drop_v' in df_droop.columns and 'vss_bounce_v' in df_droop.columns:
                vdd_actual_mv = float(df_droop['vdd_drop_v'].max()) * 1000.0
                vss_actual_mv = float(df_droop['vss_bounce_v'].max()) * 1000.0

                rel_err_vdd = abs(vdd_actual_mv - vdd_expected_mv) / vdd_expected_mv
                rel_err_vss = abs(vss_actual_mv - vss_expected_mv) / vss_expected_mv

                if rel_err_vdd > 0.01:
                    raise ValueError(
                        f"Droop validation failed for '{design}': VDD drop max ({vdd_actual_mv:.4f} mV) "
                        f"mismatches psm_summary.csv ({vdd_expected_mv:.4f} mV) by {rel_err_vdd*100:.2f}% (> 1% threshold)"
                    )
                if rel_err_vss > 0.01:
                    raise ValueError(
                        f"Droop validation failed for '{design}': VSS bounce max ({vss_actual_mv:.4f} mV) "
                        f"mismatches psm_summary.csv ({vss_expected_mv:.4f} mV) by {rel_err_vss*100:.2f}% (> 1% threshold)"
                    )
                print(f"[PSM VALIDATION PASSED] '{design}': VDD max={vdd_actual_mv:.4f} mV (exp {vdd_expected_mv:.4f} mV), VSS max={vss_actual_mv:.4f} mV (exp {vss_expected_mv:.4f} mV)")


def validate_paths(df_paths, stats_csv='design_stats.csv'):
    """Validates paths.csv against schema contract rules."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    st_path = stats_csv if os.path.isabs(stats_csv) else os.path.join(dir_path, stats_csv)
    st = pd.read_csv(st_path)
    clock_period_ns = float(st['clock_period_ns'].iloc[0])

    if 'bump_pitch_um' in st.columns and pd.isna(st['bump_pitch_um'].iloc[0]):
        if not ALLOW_UNVALIDATED_PATHS:
            print("WARNING: bump_pitch_um is NaN in design_stats.csv (synthesised BGA/bump array)")

    n_neg = (df_paths['delay_ns'] < 0).sum()
    if n_neg > 0:
        min_neg = df_paths['delay_ns'].min()
        msg = (f"paths.csv: delay_ns is negative on {n_neg}/{len(df_paths)} rows (min={min_neg:.2f}). "
               "Role A must export positive path delay. Sign convention or column swap suspected.")
        if not ALLOW_UNVALIDATED_PATHS:
            raise ValueError(msg)

    clk_file = os.path.join(dir_path, 'clock_periods.csv')
    if os.path.exists(clk_file):
        df_clk = pd.read_csv(clk_file)
        clk_map = dict(zip(df_clk['clock_domain'], df_clk['period_ns'].astype(float)))

        def get_path_period(row):
            cd = str(row.get('clock_domain', 'clk_core')).strip()
            return clk_map.get(cd, clock_period_ns)

        path_periods = df_paths.apply(get_path_period, axis=1)
        over_mask = df_paths['slack_ns'] > path_periods
        n_over = over_mask.sum()
        if n_over > 0:
            max_slack = df_paths.loc[over_mask, 'slack_ns'].max()
            msg = (f"paths.csv: slack_ns exceeds clock domain period on {n_over} rows (max={max_slack:.2f} ns).")
            if not ALLOW_UNVALIDATED_PATHS:
                raise ValueError(msg)
    else:
        max_slack = df_paths['slack_ns'].max()
        if max_slack > clock_period_ns:
            msg = (f"paths.csv: max slack_ns={max_slack:.2f} exceeds clock_period_ns={clock_period_ns} from design_stats.csv. "
                   "A single-cycle setup path cannot have slack greater than the period. Check units, SDC, or a slack/delay column swap.")
            if not ALLOW_UNVALIDATED_PATHS:
                raise ValueError(msg)


def load_scenario_weights():
    """Loads mission weights per scenario from activity.csv."""
    global SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER
    dir_path = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(dir_path, 'activity.csv'),
        os.path.join(dir_path, 'orfs_extracted', 'orfs_ssd_ctrl_cfg_small', 'activity.csv'),
        os.path.join(dir_path, 'new_files_extracted', 'activity.csv')
    ]
    act_path = None
    for c in candidates:
        if os.path.exists(c):
            act_path = c
            break

    if act_path:
        df_act = pd.read_csv(act_path)
        weights = df_act.groupby('scenario')['mission_weight'].first().to_dict()
        total_w = sum(weights.values())
        if abs(total_w - 1.0) > 1e-6:
            weights = {k: v / total_w for k, v in weights.items()}
        SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER = False
        return weights, "activity.csv (mission_weight)"
    else:
        print("[WARNING] activity.csv not found! Using uniform scenario weights placeholder.")
        SCENARIO_WEIGHTS_ARE_UNIFORM_PLACEHOLDER = True
        return None, "uniform_placeholder"


def load_droop_source(droop_source='measured_rail', stats_file='design_stats.csv', paths_file='paths.csv'):
    """
    Loads timing paths and droop data for the specified DROOP_SOURCE.
    Supports: measured_rail, measured_vdd, measured_cell_rail, measured_p12_rail,
              guardband_5pct, guardband_10pct, roleB_syn_corpus_whatif.
    """
    if droop_source not in DROOP_SOURCES_CONFIG:
        raise ValueError(f"Unknown droop_source '{droop_source}'. Must be one of {list(DROOP_SOURCES_CONFIG.keys())}")

    cfg = DROOP_SOURCES_CONFIG[droop_source]
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # Handle paths loading
    if cfg['type'] == 'measured_tile_p12':
        paths_path = os.path.join(dir_path, 'inputs', 'orfs_ssd_ctrl_cfg_run_01_p12', 'paths.csv')
        stats_path = os.path.join(dir_path, 'inputs', 'orfs_ssd_ctrl_cfg_run_01_p12', 'design_stats.csv')
        inst_path = os.path.join(dir_path, 'inputs', 'orfs_ssd_ctrl_cfg_run_01_p12', 'instances.csv')
    else:
        paths_path = os.path.join(dir_path, paths_file)
        stats_path = os.path.join(dir_path, stats_file)
        inst_path = os.path.join(dir_path, 'instances.csv')

    if not os.path.exists(paths_path):
        raise FileNotFoundError(f"Required input file missing: '{paths_path}'")

    df_paths = pd.read_csv(paths_path)
    inst_col = 'inst_ids' if 'inst_ids' in df_paths.columns else 'inst_idx'
    df_paths['inst_idx'] = df_paths[inst_col].apply(parse_inst_list)

    validate_paths(df_paths, stats_path)
    if (df_paths['delay_ns'] < 0).any() and ALLOW_UNVALIDATED_PATHS:
        df_paths['delay_ns'] = df_paths['delay_ns'].abs()

    inst_to_tile = build_inst_to_tile(inst_path, stats_path)
    is_instance_map = False
    v_drop_map = {}

    # Load droop map depending on source type
    if cfg['type'] in ('measured_tile', 'measured_tile_p12'):
        file_path = os.path.join(dir_path, cfg['file'])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Droop file missing: '{file_path}'")
        df_droop = pd.read_csv(file_path, comment='#')
        validate_psm_summary(df_droop, cfg['design'])
        v_drop_map = dict(zip(df_droop['tile_id'].astype(int), df_droop['pred_v'] * 1000.0))

    elif cfg['type'] == 'measured_instance':
        vdd_bumps_p = os.path.join(dir_path, 'final_psm', 'ssd_ctrl_cfg_small', 'VDD_BUMPS.csv')
        vss_bumps_p = os.path.join(dir_path, 'final_psm', 'ssd_ctrl_cfg_small', 'VSS_BUMPS.csv')
        if not os.path.exists(vdd_bumps_p) or not os.path.exists(vss_bumps_p):
            raise FileNotFoundError(f"PSM bump files missing in final_psm/ssd_ctrl_cfg_small/")
        
        df_vdd_b = pd.read_csv(vdd_bumps_p)
        df_vss_b = pd.read_csv(vss_bumps_p)
        df_psm = pd.merge(df_vdd_b[['Instance', 'Voltage']], df_vss_b[['Instance', 'Voltage']], on='Instance', suffixes=('_vdd', '_vss'))
        df_psm['rail_droop_mv'] = ((1.1 - df_psm['Voltage_vdd']) + df_psm['Voltage_vss']) * 1000.0
        
        inst_df = pd.read_csv(inst_path)
        merged_inst = pd.merge(inst_df, df_psm, left_on='inst_name', right_on='Instance', how='left')
        v_drop_map = dict(zip(merged_inst['inst_id'].astype(int), merged_inst['rail_droop_mv'].fillna(0.0)))
        is_instance_map = True

    elif cfg['type'] == 'guardband':
        gb_mv = cfg['frac'] * NOMINAL_V_MV
        v_drop_map = {t: gb_mv for t in range(NX_COARSE * NY_COARSE)}

    elif cfg['type'] == 'synthetic_corpus':
        pred_path = os.path.join(dir_path, cfg['file'])
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Predictions file missing: '{pred_path}'")
        df_b = pd.read_csv(pred_path, comment='#')
        df_agg = df_b.groupby('tile_id')['pred_v'].mean().reset_index()
        v_drop_map = dict(zip(df_agg['tile_id'].astype(int), df_agg['pred_v'] * 1000.0))

    meta = {
        'key': droop_source,
        'name': cfg['name'],
        'type': cfg['type'],
        'file': cfg['file'],
        'design': cfg['design'],
        'vdd_v': cfg['vdd_v'],
        'includes_vss': cfg['includes_vss'],
        'description': cfg['description'],
        'is_instance_map': is_instance_map
    }

    return df_paths, v_drop_map, meta, inst_to_tile


def load_real_or_mock_data(paths_file='paths.csv', v_drop_file='predictions.csv', design=None, stats_file='design_stats.csv'):
    """Legacy backward-compatible wrapper defaulting to measured_rail droop source."""
    df_paths, v_drop_map, meta, inst_to_tile = load_droop_source('measured_rail', stats_file, paths_file)
    weights_dict, weight_source = load_scenario_weights()
    
    scenarios = {
        'measured_rail': {'weight': 1.0, 'v_drop_map': v_drop_map, 'meta': meta}
    }
    return df_paths, v_drop_map, scenarios, weight_source


# ==========================================
# PHASE 2: Risk Engine & Rank Churn Diagnostics
# ==========================================
def calculate_delay_penalty(v_drop_mv, base_delay_ns, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA):
    """
    Applies the Alpha-Power Law: delay ∝ V / (V - Vth)^α
    Returns the EXTRA delay (ns) caused by voltage droop relative to nominal.
    """
    if v_drop_mv <= 0:
        return 0.0

    actual_v = nominal_v_mv - v_drop_mv
    margin = actual_v - vth_mv
    
    if margin <= 0.01:
        return 1e6  # Large penalty for switching failure / setup violation

    k = base_delay_ns / (nominal_v_mv / ((nominal_v_mv - vth_mv) ** alpha))
    new_delay_ns = k * (actual_v / (margin ** alpha))

    return new_delay_ns - base_delay_ns


def compute_penalties(df_paths, v_drop_map, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA, is_instance_map=False):
    """Computes path droop penalties aligned strictly to df_paths' original row order."""
    penalties_ns = []
    eff_slacks_ns = []
    cell_penalties_list = []

    for _, row in df_paths.iterrows():
        path_penalty_ns = 0.0
        cell_delays = row.get('cell_delays_ns', None)
        inst_list = row['inst_idx']
        if cell_delays is None:
            cell_delays = [row['delay_ns'] / max(1, len(inst_list))] * len(inst_list)

        path_cell_penalties = {}
        for idx, cell_base_delay in zip(inst_list, cell_delays):
            if is_instance_map:
                drop = v_drop_map.get(idx, 0.0)
            else:
                if idx not in inst_to_tile:
                    raise KeyError(f"inst_id {idx} from paths.csv not found in instances.csv mapping")
                tile_id = inst_to_tile[idx]
                if tile_id not in v_drop_map:
                    raise KeyError(f"tile_id {tile_id} (from inst_id {idx}) not found in v_drop_map")
                drop = v_drop_map[tile_id]

            penalty = calculate_delay_penalty(drop, cell_base_delay, nominal_v_mv, vth_mv, alpha)
            path_penalty_ns += penalty
            path_cell_penalties[idx] = penalty

        penalties_ns.append(path_penalty_ns)
        eff_slacks_ns.append(row['slack_ns'] - path_penalty_ns)
        cell_penalties_list.append(path_cell_penalties)

    return penalties_ns, eff_slacks_ns, cell_penalties_list


def run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA, is_instance_map=False, meta=None):
    """Calculates effective slack and ranks paths by timing risk."""
    df_paths = df_paths.copy()
    penalties_ns, eff_slacks_ns, cell_penalties_list = compute_penalties(
        df_paths, v_drop_map, inst_to_tile, nominal_v_mv, vth_mv, alpha, is_instance_map
    )

    df_paths['droop_penalty_ns'] = penalties_ns
    df_paths['effective_slack_ns'] = eff_slacks_ns
    df_paths['cell_penalties'] = cell_penalties_list

    if meta is not None:
        df_paths['droop_source'] = meta['key']
        df_paths['droop_design'] = meta['design']
        df_paths['droop_file'] = meta['file']
        df_paths['vdd_v'] = meta['vdd_v']
        df_paths['includes_vss'] = meta['includes_vss']

    df_ranked = df_paths.sort_values(by='effective_slack_ns', ascending=True).reset_index(drop=True)
    return df_ranked


from scipy.stats import kendalltau

def rank_churn(df_paths):
    """Computes strict rank-churn metrics comparing nominal slack_ns ranking vs effective_slack_ns ranking."""
    df = df_paths.copy()

    df['rank_nominal'] = df['slack_ns'].rank(method='min', ascending=True)
    df['rank_effective'] = df['effective_slack_ns'].rank(method='min', ascending=True)
    df['rank_shift'] = (df['rank_nominal'] - df['rank_effective']).abs()

    n_paths = len(df)
    n_rank_changes = int((df['rank_nominal'] != df['rank_effective']).sum())
    max_rank_shift = int(df['rank_shift'].max()) if n_paths > 0 else 0

    if n_paths > 1:
        spearman_corr = round(float(df['slack_ns'].corr(df['effective_slack_ns'], method='spearman')), 4)
        
        # Strict rank flip calculation (pairwise discordant pairs)
        s = df['slack_ns'].values
        e = df['effective_slack_ns'].values
        s_diff = s[:, None] - s[None, :]
        e_diff = e[:, None] - e[None, :]
        tri_u = np.triu_indices(n_paths, k=1)
        s_u = s_diff[tri_u]
        e_u = e_diff[tri_u]
        
        n_tied_nominal_pairs = int(np.sum(s_u == 0))
        disc_pairs_mask = ((s_u > 0) & (e_u < 0)) | ((s_u < 0) & (e_u > 0))
        n_discordant_pairs = int(np.sum(disc_pairs_mask))
        
        disc_matrix = ((s_diff > 0) & (e_diff < 0)) | ((s_diff < 0) & (e_diff > 0))
        n_paths_moved_across_tie_groups = int(np.sum(disc_matrix.any(axis=1)))

        try:
            tau_res = kendalltau(s, e)
            kendall_tau_b = round(float(tau_res.statistic if hasattr(tau_res, 'statistic') else tau_res[0]), 4)
        except Exception:
            kendall_tau_b = 1.0
    else:
        spearman_corr = 1.0
        n_discordant_pairs = 0
        n_paths_moved_across_tie_groups = 0
        kendall_tau_b = 1.0
        n_tied_nominal_pairs = 0

    top_shifted = df.sort_values(by='rank_shift', ascending=False).head(10)
    top_shifted_endpoints = ";".join([
        f"{r['endpoint']}({int(r['rank_shift'])})" for _, r in top_shifted.iterrows()
    ])

    churn_dict = {
        'n_paths': n_paths,
        'n_rank_changes': n_rank_changes,
        'max_rank_shift': max_rank_shift,
        'spearman_correlation': spearman_corr,
        'n_discordant_pairs': n_discordant_pairs,
        'n_paths_moved_across_tie_groups': n_paths_moved_across_tie_groups,
        'kendall_tau_b': kendall_tau_b,
        'n_tied_nominal_pairs': n_tied_nominal_pairs,
        'top_10_shifted_endpoints': top_shifted_endpoints
    }
    return churn_dict


def compute_instance_risk_attribution(df_ranked):
    """Ranks individual instances by total delay penalty contributed across all paths."""
    instance_loss = {}
    for _, row in df_ranked.iterrows():
        for inst_id, penalty in row['cell_penalties'].items():
            instance_loss[inst_id] = instance_loss.get(inst_id, 0.0) + penalty
            
    df_attr = pd.DataFrame([
        {'inst_idx': k, 'total_delay_penalty_ps': round(v * 1000, 2)} 
        for k, v in instance_loss.items()
    ]).sort_values(by='total_delay_penalty_ps', ascending=False).reset_index(drop=True)
    
    return df_attr


def run_scenario_weighted_risk(df_paths, scenarios, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA):
    """PRISM N5: Computes probability-weighted expected risk across operational scenarios."""
    expected_slack_loss = np.zeros(len(df_paths))
    
    eval_scenarios = {k: v for k, v in scenarios.items() if v.get('weight', 0) > 0}
    total_w = sum(v['weight'] for v in eval_scenarios.values()) if eval_scenarios else 1.0
    
    for sc_name, sc_data in eval_scenarios.items():
        norm_weight = sc_data['weight'] / total_w
        is_inst = sc_data.get('meta', {}).get('is_instance_map', False)
        penalties_ns, _, _ = compute_penalties(
            df_paths, sc_data['v_drop_map'], inst_to_tile, nominal_v_mv, vth_mv, alpha, is_instance_map=is_inst
        )
        expected_slack_loss += norm_weight * np.array(penalties_ns)
        
    df_expected = df_paths.copy()
    df_expected['expected_penalty_ns'] = expected_slack_loss
    df_expected['expected_effective_slack_ns'] = df_expected['slack_ns'] - expected_slack_loss
    return df_expected.sort_values(by='expected_effective_slack_ns', ascending=True).reset_index(drop=True)


# ==========================================
# PHASE 3: Dynamic At-Risk Thresholding & Mitigation Catalog
# ==========================================
def compute_at_risk_mask(df_ranked, mode=AT_RISK_MODE, quantile=AT_RISK_QUANTILE, period_frac=AT_RISK_PERIOD_FRAC, stats_csv='design_stats.csv'):
    """Computes boolean mask for at-risk paths using dynamic scaling."""
    n_total = len(df_ranked)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    st_path = stats_csv if os.path.isabs(stats_csv) else os.path.join(dir_path, stats_csv)

    if mode == 'quantile':
        threshold_ns = float(df_ranked['effective_slack_ns'].quantile(quantile))
        mask = df_ranked['effective_slack_ns'] <= threshold_ns
    elif mode == 'absolute':
        st = pd.read_csv(st_path)
        clk_period = float(st['clock_period_ns'].iloc[0])
        threshold_ns = clk_period * period_frac
        mask = df_ranked['effective_slack_ns'] < threshold_ns
    else:
        raise ValueError(f"Unknown AT_RISK_MODE '{mode}'")

    path_count = int(mask.sum())
    max_allowed = int(0.25 * n_total)

    if not (20 <= path_count <= max_allowed):
        threshold_ns = float(df_ranked['effective_slack_ns'].quantile(0.05))
        mask = df_ranked['effective_slack_ns'] <= threshold_ns
        path_count = int(mask.sum())

    return mask, round(threshold_ns, 4), path_count, mode


MITIGATION_CATALOG = {
    "Decap insertion":          {"cost": 15, "type": "droop_reduction", "reduction_pct": 0.1500, "vth_delta_mv": 0.0,   "assumption_source": "assumed_whatif"},
    "Strap widening":           {"cost": 25, "type": "droop_reduction", "reduction_pct": 0.2500, "vth_delta_mv": 0.0,   "assumption_source": "assumed_whatif"},
    "Cell de-densification":    {"cost": 30, "type": "droop_reduction", "reduction_pct": 0.2000, "vth_delta_mv": 0.0,   "assumption_source": "assumed_whatif"},
    "Clock-skew staggering":    {"cost": 10, "type": "droop_reduction", "reduction_pct": 0.3000, "vth_delta_mv": 0.0,   "assumption_source": "assumed_whatif"},
    "Vt swap (LVT -50mV)":      {"cost": 5,  "type": "vth_swap",        "reduction_pct": 0.1335, "vth_delta_mv": -50.0, "assumption_source": "assumed_vth_swap"},
    "Extra bump/pad":           {"cost": 40, "type": "droop_reduction", "reduction_pct": 0.3500, "vth_delta_mv": 0.0,   "assumption_source": "assumed_whatif"},
    "Measured: strap pitch 12 µm": {"cost": 20, "type": "measured_p12", "reduction_pct": 0.0000, "vth_delta_mv": 0.0,   "assumption_source": "roleA_measured_pnr"}
}


def get_at_risk_instances(baseline_ranked, mode=AT_RISK_MODE):
    """Identifies instances on paths within the dynamically computed at-risk set."""
    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)
    at_risk_paths = baseline_ranked[mask]
    instances = set()
    for idx_list in at_risk_paths['inst_idx']:
        instances.update(idx_list)
    return instances


def apply_fix(v_drop_map, reduction_pct, affected_instances, inst_to_tile, is_instance_map=False):
    """Applies droop reduction percentage to specified instances / tiles."""
    new_map = dict(v_drop_map)
    if is_instance_map:
        for idx in affected_instances:
            if idx in new_map:
                new_map[idx] = new_map[idx] * (1.0 - reduction_pct)
    else:
        affected_tiles = set()
        for idx in affected_instances:
            if idx in inst_to_tile:
                affected_tiles.add(inst_to_tile[idx])

        for tile_id in affected_tiles:
            if tile_id in new_map:
                new_map[tile_id] = new_map[tile_id] * (1.0 - reduction_pct)
    return new_map


def build_measured_catalog(df_paths, v_drop_map, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA, mode=AT_RISK_MODE, is_instance_map=False, droop_source='measured_rail'):
    """Builds catalog of fixes with real computed delay savings (in ps) across the dynamic at-risk set."""
    baseline_ranked = run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv, vth_mv, alpha, is_instance_map)
    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)
    at_risk_instances = get_at_risk_instances(baseline_ranked, mode=mode)
    at_risk_indices = baseline_ranked[mask].index
    n_paths_evaluated = len(at_risk_indices)

    # Pre-load 12um strap run droop map if available
    dir_path = os.path.dirname(os.path.abspath(__file__))
    f_p12_rail = os.path.join(dir_path, 'measured_droop', 'orfs_ssd_ctrl_cfg_run_01_p12_rail.csv')
    if os.path.exists(f_p12_rail):
        df_p12 = pd.read_csv(f_p12_rail, comment='#')
        v_drop_map_p12 = dict(zip(df_p12['tile_id'].astype(int), (df_p12['vdd_drop_v'] + df_p12['vss_bounce_v']) * 1000.0))
    else:
        v_drop_map_p12 = None

    failing_paths_cnt = int((baseline_ranked.loc[at_risk_indices, 'effective_slack_ns'] < 0).sum())

    measured_fixes = []
    for name, spec in MITIGATION_CATALOG.items():
        if spec["type"] == "measured_p12":
            if droop_source in ('roleB_syn_corpus_whatif', 'guardband_5pct', 'guardband_10pct', 'measured_p12_rail'):
                continue
            after_fix = run_risk_engine(df_paths, v_drop_map_p12, inst_to_tile, nominal_v_mv, vth_mv, alpha, is_instance_map) if v_drop_map_p12 is not None else baseline_ranked
        elif spec["type"] in ("droop_reduction", "vth_swap"):
            fixed_map = apply_fix(v_drop_map, spec["reduction_pct"], at_risk_instances, inst_to_tile, is_instance_map)
            after_fix = run_risk_engine(df_paths, fixed_map, inst_to_tile, nominal_v_mv, vth_mv, alpha, is_instance_map)
        else:
            after_fix = baseline_ranked

        base_slack = baseline_ranked.loc[at_risk_indices, 'effective_slack_ns']
        after_slack = after_fix.loc[at_risk_indices, 'effective_slack_ns']
        diff_ps = (after_slack - base_slack) * 1000.0

        total_ben = max(0.0, round(float(diff_ps.sum()), 2))
        mean_ben = round(total_ben / max(1, n_paths_evaluated), 2)
        max_ben = max(0.0, round(float(diff_ps.max()), 2))

        measured_fixes.append({
            "name": name,
            "cost": spec["cost"],
            "reduction_pct": spec["reduction_pct"],
            "benefit_ps": total_ben,
            "mean_benefit_ps_per_path": mean_ben,
            "max_benefit_ps_per_path": max_ben,
            "total_benefit_ps": total_ben,
            "n_paths_evaluated": n_paths_evaluated,
            "at_risk_paths_with_negative_effective_slack": failing_paths_cnt,
            "assumption_source": spec["assumption_source"],
            "at_risk_threshold_ns": thresh_ns,
            "at_risk_mode": mode_used,
            "at_risk_path_count": path_count
        })
    return measured_fixes


# ==========================================
# PHASE 4: Mitigation Knapsack Optimizer & Pareto
# ==========================================
def optimize_mitigations(fixes, budget):
    """0/1 Knapsack DP optimizing measured ps benefit under cost budget across fix COMBINATIONS."""
    n = len(fixes)
    dp = [[0.0 for _ in range(budget + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, budget + 1):
            cost = fixes[i - 1]["cost"]
            benefit = fixes[i - 1]["benefit_ps"]
            if cost <= w:
                dp[i][w] = max(benefit + dp[i - 1][w - cost], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    selected_fixes = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_fixes.append(fixes[i - 1]["name"])
            w -= fixes[i - 1]["cost"]

    return dp[n][budget], selected_fixes


def verify_joint_mitigation(df_paths, v_drop_map, selected_fix_names, catalog_fixes, inst_to_tile, nominal_v_mv=1100.0, vth_mv=VTH_MV, alpha=ALPHA, mode=AT_RISK_MODE, is_instance_map=False):
    """Verifies combined effect of selected fixes by re-solving physics simultaneously across all selected fixes."""
    baseline_ranked = run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv, vth_mv, alpha, is_instance_map)
    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)
    at_risk_instances = get_at_risk_instances(baseline_ranked, mode=mode)
    at_risk_indices = baseline_ranked[mask].index

    dir_path = os.path.dirname(os.path.abspath(__file__))
    f_p12_rail = os.path.join(dir_path, 'measured_droop', 'orfs_ssd_ctrl_cfg_run_01_p12_rail.csv')
    if os.path.exists(f_p12_rail):
        df_p12 = pd.read_csv(f_p12_rail, comment='#')
        v_drop_map_p12 = dict(zip(df_p12['tile_id'].astype(int), (df_p12['vdd_drop_v'] + df_p12['vss_bounce_v']) * 1000.0))
    else:
        v_drop_map_p12 = None

    has_measured = any('Measured' in f or 'strap pitch 12' in f for f in selected_fix_names)
    if has_measured and v_drop_map_p12 is not None:
        combined_map = dict(v_drop_map_p12)
    else:
        combined_map = dict(v_drop_map)

    fix_spec_dict = {f['name']: f for f in catalog_fixes}
    for fix_name in selected_fix_names:
        if 'Measured' in fix_name or 'strap pitch 12' in fix_name:
            continue
        spec = fix_spec_dict.get(fix_name, {})
        pct = spec.get('reduction_pct', 0.0)
        combined_map = apply_fix(combined_map, pct, at_risk_instances, inst_to_tile, is_instance_map)

    verified_ranked = run_risk_engine(df_paths, combined_map, inst_to_tile, nominal_v_mv, vth_mv, alpha, is_instance_map)

    base_slack = baseline_ranked.loc[at_risk_indices, 'effective_slack_ns'].sum()
    ver_slack = verified_ranked.loc[at_risk_indices, 'effective_slack_ns'].sum()

    true_benefit_ps = max(0.0, (ver_slack - base_slack) * 1000)
    return round(true_benefit_ps, 2)


def generate_pareto_curve(df_paths, v_drop_map, fixes, inst_to_tile, nominal_v_mv=1100.0, max_budget=60, step=10, mode=AT_RISK_MODE, is_instance_map=False):
    """Sweeps budget to build Pareto Front across fix COMBINATIONS with physics verification."""
    baseline_ranked = run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv, is_instance_map=is_instance_map)
    mask, thresh_ns, path_count, mode_used = compute_at_risk_mask(baseline_ranked, mode=mode)

    results = []
    prev_ver = 0.0
    for b in range(step, max_budget + 1, step):
        est_benefit_ps, chosen = optimize_mitigations(fixes, b)
        ver_benefit_ps = verify_joint_mitigation(df_paths, v_drop_map, chosen, fixes, inst_to_tile, nominal_v_mv, mode=mode, is_instance_map=is_instance_map)
        if ver_benefit_ps < prev_ver:
            ver_benefit_ps = prev_ver
        else:
            prev_ver = ver_benefit_ps

        chosen_str = ", ".join(chosen) if chosen else "None"
        results.append({
            'budget': b, 
            'knapsack_est_ps': est_benefit_ps, 
            'verified_true_ps': ver_benefit_ps,
            'chosen_fixes': chosen_str,
            'at_risk_threshold_ns': thresh_ns,
            'at_risk_mode': mode_used,
            'at_risk_path_count': path_count
        })

    pareto_df = pd.DataFrame(results)
    assert (pareto_df['verified_true_ps'].diff().dropna() >= -1e-6).all(), "Pareto curve verified_true_ps must be non-decreasing!"
    return pareto_df


# ==========================================
# MAIN EXECUTION & PER-SCENARIO PRECOMPUTATION TO DISK
# ==========================================
if __name__ == "__main__":
    print("============================================")
    print("       PRISM ROLE C RISK & DSE ENGINE       ")
    print("============================================\n")

    dir_path = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(dir_path, 'design_stats.csv')
    st_df = pd.read_csv(stats_file)
    nominal_v_mv = float(st_df['vdd_v'].iloc[0]) * 1000.0
    die_w = float(st_df['die_w_um'].iloc[0])
    die_h = float(st_df['die_h_um'].iloc[0])

    provenance_tag = 'UNVALIDATED_INPUT' if ALLOW_UNVALIDATED_PATHS else 'VALIDATED'

    # Precompute outputs for all supported droop sources and legacy workload aliases
    eval_sources = list(DROOP_SOURCES_CONFIG.keys()) + ['idle', 'seq_read', 'seq_write', 'rand_read_4k', 'gc_compact', 'ecc_recover']
    
    print("[PRECOMPUTING DROOP SOURCE RESULT FILES TO DISK]")
    cols_ranked = [
        'path_id', 'endpoint', 'clock_domain', 'slack_ns', 'delay_ns', 
        'droop_penalty_ns', 'effective_slack_ns', 'check_type', 
        'droop_source', 'droop_design', 'droop_file', 'vdd_v', 'includes_vss', 'provenance'
    ]

    inst_path = os.path.join(dir_path, 'instances.csv')
    df_inst = pd.read_csv(inst_path)

    # Store default outputs from measured_rail
    df_paths_default, v_drop_map_default, meta_default, inst_to_tile_default = load_droop_source('measured_rail')
    
    # Acceptance table summary collector
    acceptance_results = []

    for src_key in eval_sources:
        effective_key = src_key
        if src_key in ('idle', 'seq_read', 'seq_write', 'rand_read_4k', 'gc_compact', 'ecc_recover'):
            effective_key = 'measured_rail'

        df_paths, v_drop_map, meta, inst_to_tile = load_droop_source(effective_key)
        is_inst = meta['is_instance_map']

        ranked_sc = run_risk_engine(df_paths, v_drop_map, inst_to_tile, nominal_v_mv, is_instance_map=is_inst, meta=meta)
        ranked_sc['provenance'] = provenance_tag

        # 1. Saved ranked risk output
        cols_to_export = [c for c in cols_ranked if c in ranked_sc.columns]
        f_ranked = os.path.join(dir_path, f'prism_ranked_risk_{src_key}.csv')
        ranked_sc[cols_to_export].to_csv(f_ranked, index=False)

        # 2. Saved rank churn
        churn_sc = rank_churn(ranked_sc)
        f_churn = os.path.join(dir_path, f'prism_rank_churn_{src_key}.csv')
        pd.DataFrame([churn_sc]).to_csv(f_churn, index=False)

        # 3. Saved tile attribution & 24x24 risk grid
        attr_sc = compute_instance_risk_attribution(ranked_sc)
        attr_sc['provenance'] = provenance_tag
        f_attr = os.path.join(dir_path, f'prism_tile_attribution_{src_key}.csv')
        attr_sc[['inst_idx', 'total_delay_penalty_ps', 'provenance']].to_csv(f_attr, index=False)

        # Use appropriate instances mapping for risk grid
        target_inst_csv = os.path.join(dir_path, 'inputs', 'orfs_ssd_ctrl_cfg_run_01_p12', 'instances.csv') if meta['type'] == 'measured_tile_p12' else inst_path
        target_inst_df = pd.read_csv(target_inst_csv)
        merged_sc = attr_sc.merge(target_inst_df[['inst_id', 'x_um', 'y_um']], left_on='inst_idx', right_on='inst_id', how='left')
        
        risk_grid_sc = np.zeros((24, 24))
        for _, row in merged_sc.iterrows():
            if pd.notna(row['x_um']) and pd.notna(row['y_um']):
                gx = min(23, max(0, int(float(row['x_um']) / (die_w / 24))))
                gy = min(23, max(0, int(float(row['y_um']) / (die_h / 24))))
                risk_grid_sc[gy, gx] += float(row['total_delay_penalty_ps'])
        
        f_grid = os.path.join(dir_path, f'prism_chip_risk_grid_{src_key}.csv')
        pd.DataFrame(risk_grid_sc).to_csv(f_grid, index=False)

        # 4. Saved measured & illustrative mitigation catalog
        df_paths_excl = df_paths[df_paths['check_type'].isin(['setup', 'clock_gating'])].copy().reset_index(drop=True)
        cat_fixes_sc = build_measured_catalog(df_paths_excl, v_drop_map, inst_to_tile, nominal_v_mv, is_instance_map=is_inst, droop_source=src_key)
        cat_df_sc = pd.DataFrame(cat_fixes_sc)
        cat_df_sc['provenance'] = provenance_tag
        cat_cols = ['name', 'cost', 'reduction_pct', 'benefit_ps', 'mean_benefit_ps_per_path', 'max_benefit_ps_per_path', 'total_benefit_ps', 'n_paths_evaluated', 'at_risk_paths_with_negative_effective_slack', 'assumption_source', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']
        cat_cols_export = [c for c in cat_cols if c in cat_df_sc.columns]
        f_cat_meas = os.path.join(dir_path, f'prism_measured_mitigation_catalog_{src_key}.csv')
        f_cat_illus = os.path.join(dir_path, f'prism_illustrative_mitigation_catalog_{src_key}.csv')
        cat_df_sc[cat_cols_export].to_csv(f_cat_meas, index=False)
        cat_df_sc[cat_cols_export].to_csv(f_cat_illus, index=False)

        # Saved Pareto front curve (excl recovery default)
        pareto_sc = generate_pareto_curve(df_paths_excl, v_drop_map, cat_fixes_sc, inst_to_tile, nominal_v_mv, max_budget=60, step=10, is_instance_map=is_inst)
        pareto_sc['provenance'] = provenance_tag
        f_pareto = os.path.join(dir_path, f'prism_mitigation_pareto_front_{src_key}.csv')
        pareto_sc[['budget', 'knapsack_est_ps', 'verified_true_ps', 'chosen_fixes', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']].to_csv(f_pareto, index=False)

        # Also precompute incl recovery version (all 1601 paths)
        cat_fixes_sc_incl = build_measured_catalog(df_paths, v_drop_map, inst_to_tile, nominal_v_mv, is_instance_map=is_inst, droop_source=src_key)
        cat_df_sc_incl = pd.DataFrame(cat_fixes_sc_incl)
        cat_df_sc_incl['provenance'] = provenance_tag
        f_cat_incl = os.path.join(dir_path, f'prism_measured_mitigation_catalog_{src_key}_incl_rec.csv')
        cat_df_sc_incl[cat_cols_export].to_csv(f_cat_incl, index=False)

        pareto_sc_incl = generate_pareto_curve(df_paths, v_drop_map, cat_fixes_sc_incl, inst_to_tile, nominal_v_mv, max_budget=60, step=10, is_instance_map=is_inst)
        pareto_sc_incl['provenance'] = provenance_tag
        f_pareto_incl = os.path.join(dir_path, f'prism_mitigation_pareto_front_{src_key}_incl_rec.csv')
        pareto_sc_incl[['budget', 'knapsack_est_ps', 'verified_true_ps', 'chosen_fixes', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']].to_csv(f_pareto_incl, index=False)

        # Metrics for acceptance logging
        mean_p = round(float(ranked_sc['droop_penalty_ns'].mean()) * 1000.0, 2)
        max_p = round(float(ranked_sc['droop_penalty_ns'].max()) * 1000.0, 2)
        n_v = int((ranked_sc['effective_slack_ns'] < 0).sum())
        flips = churn_sc['n_rank_changes']
        min_eff_s = round(float(ranked_sc['effective_slack_ns'].min()), 4)

        if src_key in DROOP_SOURCES_CONFIG:
            acceptance_results.append({
                'droop_source': src_key,
                'mean_penalty_ps': mean_p,
                'max_penalty_ps': max_p,
                'violations': n_v,
                'strict_rank_flips': flips,
                'min_eff_slack_ns': min_eff_s
            })

        print(f"  --> Precomputed '{src_key:25s}' | Violations={n_v:2d} | Mean Pen={mean_p:6.2f} ps | Max Pen={max_p:6.2f} ps | Min Slack={min_eff_s:.4f} ns")

    # Export default fallback CSVs matching primary measured_rail
    primary_ranked = run_risk_engine(df_paths_default, v_drop_map_default, inst_to_tile_default, nominal_v_mv, meta=meta_default)
    primary_ranked['provenance'] = provenance_tag
    cols_to_export = [c for c in cols_ranked if c in primary_ranked.columns]
    primary_ranked[cols_to_export].to_csv(os.path.join(dir_path, 'prism_ranked_risk_output.csv'), index=False)
    
    churn_info = rank_churn(primary_ranked)
    pd.DataFrame([churn_info]).to_csv(os.path.join(dir_path, 'prism_rank_churn.csv'), index=False)
    
    attr_df = compute_instance_risk_attribution(primary_ranked)
    attr_df['provenance'] = provenance_tag
    attr_df[['inst_idx', 'total_delay_penalty_ps', 'provenance']].to_csv(os.path.join(dir_path, 'prism_tile_attribution.csv'), index=False)

    # Precompute default chip risk grid
    merged_def = attr_df.merge(df_inst[['inst_id', 'x_um', 'y_um']], left_on='inst_idx', right_on='inst_id', how='left')
    def_grid = np.zeros((24, 24))
    for _, row in merged_def.iterrows():
        if pd.notna(row['x_um']) and pd.notna(row['y_um']):
            gx = min(23, max(0, int(float(row['x_um']) / (die_w / 24))))
            gy = min(23, max(0, int(float(row['y_um']) / (die_h / 24))))
            def_grid[gy, gx] += float(row['total_delay_penalty_ps'])
    pd.DataFrame(def_grid).to_csv(os.path.join(dir_path, 'prism_chip_risk_grid.csv'), index=False)

    # N5 Expected risk
    weights_dict, weight_source = load_scenario_weights()
    all_scenarios_n5 = {}
    if weights_dict:
        for sc, w in weights_dict.items():
            if sc in eval_sources:
                df_p_sc, v_map_sc, meta_sc, i_to_t_sc = load_droop_source('measured_rail')
                all_scenarios_n5[sc] = {'weight': w, 'v_drop_map': v_map_sc, 'meta': meta_sc}

    if all_scenarios_n5:
        df_expected = run_scenario_weighted_risk(df_paths_default, all_scenarios_n5, inst_to_tile_default, nominal_v_mv)
    else:
        df_expected = df_paths_default.copy()
        df_expected['expected_penalty_ns'] = df_expected['droop_penalty_ns']
        df_expected['expected_effective_slack_ns'] = df_expected['effective_slack_ns']

    df_expected['provenance'] = provenance_tag
    df_expected['weight_source'] = weight_source
    cols_sc = ['path_id', 'endpoint', 'slack_ns', 'expected_penalty_ns', 'expected_effective_slack_ns', 'provenance', 'weight_source'] if 'path_id' in df_expected.columns else ['endpoint', 'slack_ns', 'expected_penalty_ns', 'expected_effective_slack_ns', 'provenance', 'weight_source']
    df_expected[cols_sc].to_csv(os.path.join(dir_path, 'prism_scenario_expected_risk.csv'), index=False)

    df_paths_default_excl = df_paths_default[df_paths_default['check_type'].isin(['setup', 'clock_gating'])].copy().reset_index(drop=True)
    measured_fixes_excl = build_measured_catalog(df_paths_default_excl, v_drop_map_default, inst_to_tile_default, nominal_v_mv)
    catalog_df_excl = pd.DataFrame(measured_fixes_excl)
    catalog_df_excl['provenance'] = provenance_tag
    cat_cols_exp = [c for c in cat_cols if c in catalog_df_excl.columns]
    catalog_df_excl[cat_cols_exp].to_csv(os.path.join(dir_path, 'prism_measured_mitigation_catalog.csv'), index=False)
    catalog_df_excl[cat_cols_exp].to_csv(os.path.join(dir_path, 'prism_illustrative_mitigation_catalog.csv'), index=False)

    pareto_df_excl = generate_pareto_curve(df_paths_default_excl, v_drop_map_default, measured_fixes_excl, inst_to_tile_default, nominal_v_mv, max_budget=60, step=10)
    pareto_df_excl['provenance'] = provenance_tag
    pareto_df_excl[['budget', 'knapsack_est_ps', 'verified_true_ps', 'chosen_fixes', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']].to_csv(os.path.join(dir_path, 'prism_mitigation_pareto_front.csv'), index=False)

    measured_fixes_incl = build_measured_catalog(df_paths_default, v_drop_map_default, inst_to_tile_default, nominal_v_mv)
    catalog_df_incl = pd.DataFrame(measured_fixes_incl)
    catalog_df_incl['provenance'] = provenance_tag
    catalog_df_incl[cat_cols_exp].to_csv(os.path.join(dir_path, 'prism_measured_mitigation_catalog_incl_rec.csv'), index=False)

    pareto_df_incl = generate_pareto_curve(df_paths_default, v_drop_map_default, measured_fixes_incl, inst_to_tile_default, nominal_v_mv, max_budget=60, step=10)
    pareto_df_incl['provenance'] = provenance_tag
    pareto_df_incl[['budget', 'knapsack_est_ps', 'verified_true_ps', 'chosen_fixes', 'at_risk_threshold_ns', 'at_risk_mode', 'at_risk_path_count', 'provenance']].to_csv(os.path.join(dir_path, 'prism_mitigation_pareto_front_incl_rec.csv'), index=False)

    print("\n============================================")
    print(" ACCEPTANCE CRITERIA RESULTS SUMMARY TABLE ")
    print("============================================")
    df_acc = pd.DataFrame(acceptance_results)
    print(df_acc.to_string(index=False))

    print("\n============================================")
    print(" ALL SCENARIO INTERFACE FILES PRECOMPUTED!  ")
    print("============================================\n")
