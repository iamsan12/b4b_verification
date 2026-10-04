import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def snap_to_grid(val, grid=[8, 12, 16, 24]):
    """Snaps a continuous float to the nearest valid physical strap pitch grid value."""
    return min(grid, key=lambda x: abs(x - val))

def select_top_candidates_bo(sweep_csv='ssd_ctrl_sweep_summary.csv', pareto_csv='tier2_pareto_front.csv', num_candidates=4):
    """
    Tier 3 Optimization / Candidate Selection:
    Fits GP on real measured PnR sweep data from ssd_ctrl_sweep_summary.csv.
    Objective: minimize worst-case IR drop (ir_drop_worst_mv).
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    sweep_path = os.path.join(dir_path, sweep_csv)
    
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(f"Required measured sweep file '{sweep_csv}' not found.")
    
    df_sweep = pd.read_csv(sweep_path)
    
    # Feature columns (strictly physical parameters, no design_id or hash)
    feature_cols = ['NUM_CH', 'DATA_W', 'ECC_LANES', 'CLK_GATE_EN', 'pdn_strap_pitch_um']
    
    # Clean pdn_strap_pitch_um (parse platform strings if present)
    def parse_pitch(val):
        if isinstance(val, (int, float)) and not np.isnan(val):
            return float(val)
        if isinstance(val, str) and 'platform' in val:
            return 30.0  # Platform default
        try:
            return float(val)
        except (ValueError, TypeError):
            return 24.0

    df_sweep['pitch_clean'] = df_sweep['pdn_strap_pitch_um'].apply(parse_pitch)
    
    X_train = df_sweep[['NUM_CH', 'DATA_W', 'ECC_LANES', 'CLK_GATE_EN', 'pitch_clean']].values
    y_train = df_sweep['ir_drop_worst_mv'].values  # Objective: minimize worst IR drop (mV)
    
    # Fit GP on real measured observations (6 observations)
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True, random_state=42)
    gp.fit(X_train, y_train)
    
    # Sort candidates strictly by measured objective: ir_drop_worst_mv ascending (minimized).
    # Tie-breaking rules: break ties by power_mw ascending, then die_area_um2 ascending.
    bo_runs_rows = df_sweep[df_sweep['bo_rank'].notna()].copy().sort_values(
        by=['ir_drop_worst_mv', 'power_mw', 'die_area_um2'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    proposals = []
    for rank_idx, row in bo_runs_rows.head(num_candidates).iterrows():
        rank_id = rank_idx + 1
        # Derive proposal_id from config column to permanently bind ID to configuration
        prop_id = str(row['config']) if 'config' in row and pd.notna(row['config']) else f"cfg_run_{int(row['bo_rank']):02d}"
        proposals.append({
            'rank': rank_id,
            'proposal_id': prop_id,
            'config_base': 'cfg_small',
            'NUM_CH': int(row['NUM_CH']),
            'DATA_W': int(row['DATA_W']),
            'ECC_LANES': int(row['ECC_LANES']),
            'CLK_GATE_EN': int(row['CLK_GATE_EN']),
            'PDN_STRAP_PITCH_UM': int(snap_to_grid(row['pitch_clean'])),
            'executed_strap_pitch_um': int(row['pitch_clean']),
            'bo_score': round(float(row['ir_drop_worst_mv']), 4),
            'method': 'heuristic_ranking_over_measured_runs',
            'rank_basis': 'measured_ir_drop_worst_mv_ascending',
            'legacy_bo_rank': int(row['bo_rank'])
        })
        
    df_proposals = pd.DataFrame(proposals)
    out_csv = os.path.join(dir_path, 'bo_proposals.csv')
    df_proposals.to_csv(out_csv, index=False)

    print(f"\n============================================")
    print(f" TIER 3 BO SELECTION COMPLETE ({len(df_proposals)} MEASURED PROPOSALS) ")
    print(f" Saved proposals to 'bo_proposals.csv'")
    print(f" Output Path: {out_csv}")
    print(f"============================================")
    print(df_proposals.to_string(index=False))

    # FIX 4 (b): Evaluate GP LCB acquisition on unobserved candidate pool from sobol_samples_tier1.csv
    sobol_path = os.path.join(dir_path, 'sobol_samples_tier1.csv')
    if os.path.exists(sobol_path):
        df_sobol = pd.read_csv(sobol_path)
        
        # Build set of executed tuples from df_sweep
        executed_combos = set()
        for _, row in df_sweep.iterrows():
            tup = (int(row['NUM_CH']), int(row['DATA_W']), int(row['ECC_LANES']), int(row['CLK_GATE_EN']), int(snap_to_grid(row['pitch_clean'])))
            executed_combos.add(tup)
            
        unobserved = []
        seen = set()
        for _, row in df_sobol.iterrows():
            ch = int(row['NUM_CH'])
            w = int(row['DATA_W'])
            ecc = int(row['ECC_LANES'])
            clk = int(np.clip(round(row['CLK_GATE_EN']), 0, 1))
            pitch = int(snap_to_grid(row['PDN_STRAP_PITCH_UM']))
            
            tup = (ch, w, ecc, clk, pitch)
            if tup in executed_combos or tup in seen:
                continue
            seen.add(tup)
            
            unobserved.append({
                'NUM_CH': ch,
                'DATA_W': w,
                'ECC_LANES': ecc,
                'CLK_GATE_EN': clk,
                'PDN_STRAP_PITCH_UM': pitch
            })
            
        if len(unobserved) > 0:
            df_unobs = pd.DataFrame(unobserved)
            X_cand = df_unobs[['NUM_CH', 'DATA_W', 'ECC_LANES', 'CLK_GATE_EN', 'PDN_STRAP_PITCH_UM']].values
            mu, sigma = gp.predict(X_cand, return_std=True)
            lcb = mu - 1.96 * sigma
            
            df_unobs['mu_mv'] = np.round(mu, 4)
            df_unobs['sigma_mv'] = np.round(sigma, 4)
            df_unobs['lcb_score_mv'] = np.round(lcb, 4)
            
            df_unobs_sorted = df_unobs.sort_values('lcb_score_mv', ascending=True).reset_index(drop=True)
            
            # Greedy diversity selection loop over normalized knob space
            def norm_vec(row):
                return np.array([
                    (float(row['NUM_CH']) - 1.0) / 7.0,
                    (float(row['DATA_W']) - 16.0) / 48.0,
                    (float(row['ECC_LANES']) - 1.0) / 3.0,
                    (float(row['CLK_GATE_EN']) - 0.0) * 0.2,
                    (float(row['PDN_STRAP_PITCH_UM']) - 8.0) / 16.0
                ])

            selected_rows = []
            selected_vecs = []
            min_dist_threshold = 0.25

            for idx, row in df_unobs_sorted.iterrows():
                vec = norm_vec(row)
                if not selected_vecs:
                    selected_rows.append(row)
                    selected_vecs.append(vec)
                else:
                    min_d = min(np.linalg.norm(vec - s_vec) for s_vec in selected_vecs)
                    if min_d >= min_dist_threshold:
                        selected_rows.append(row)
                        selected_vecs.append(vec)
                if len(selected_rows) == num_candidates:
                    break

            next_proposals = []
            for rank_idx, row in enumerate(selected_rows):
                rank_id = rank_idx + 1
                next_proposals.append({
                    'rank': rank_id,
                    'candidate_id': f"cfg_next_{rank_id:02d}",
                    'config_base': 'cfg_small',
                    'NUM_CH': int(row['NUM_CH']),
                    'DATA_W': int(row['DATA_W']),
                    'ECC_LANES': int(row['ECC_LANES']),
                    'CLK_GATE_EN': int(row['CLK_GATE_EN']),
                    'PDN_STRAP_PITCH_UM': int(row['PDN_STRAP_PITCH_UM']),
                    'mu_mv': float(row['mu_mv']),
                    'sigma_mv': float(row['sigma_mv']),
                    'lcb_score_mv': float(row['lcb_score_mv']),
                    'method': 'gp_lcb_unobserved_diverse',
                    'n_training_obs': len(df_sweep),
                    'training_note': 'Fitted on 6 measured PnR observations from ssd_ctrl_sweep_summary.csv (small training set, diverse acquisition)'
                })
                
            df_next = pd.DataFrame(next_proposals)
            out_next_csv = os.path.join(dir_path, 'bo_next_candidates.csv')
            df_next.to_csv(out_next_csv, index=False)
            print(f"\nSaved {len(df_next)} unobserved BO candidate proposals to 'bo_next_candidates.csv'")
            print(df_next.to_string(index=False))

    return df_proposals

if __name__ == "__main__":
    select_top_candidates_bo()


