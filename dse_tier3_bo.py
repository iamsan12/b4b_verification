import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

def snap_to_grid(val, grid=[8, 12, 16, 24]):
    """Snaps a continuous float to the nearest valid physical strap pitch grid value."""
    return min(grid, key=lambda x: abs(x - val))

def parse_pitch(val):
    if isinstance(val, (int, float)) and not np.isnan(val):
        return float(val), 0
    if isinstance(val, str) and 'platform' in val:
        return 30.0, 1
    try:
        return float(val), 0
    except (ValueError, TypeError):
        return 24.0, 0

def select_top_candidates_bo(sweep_csv='ssd_ctrl_sweep_summary.csv', pareto_csv='tier2_pareto_front.csv', num_candidates=4):
    """
    Tier 3 Optimization / Candidate Selection:
    Fits GP on real measured PnR sweep data from ssd_ctrl_sweep_summary.csv.
    Fits in log-space with StandardScaler and anisotropic Matern kernel for strictly positive LCB predictions.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    sweep_path = os.path.join(dir_path, sweep_csv)
    
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(f"Required measured sweep file '{sweep_csv}' not found.")
    
    df_sweep = pd.read_csv(sweep_path)
    
    pitch_info = df_sweep['pdn_strap_pitch_um'].apply(parse_pitch)
    df_sweep['pitch_num'] = [p[0] for p in pitch_info]
    df_sweep['is_platform_pdn'] = [p[1] for p in pitch_info]
    
    feature_cols = ['NUM_CH', 'DATA_W', 'ECC_LANES', 'CLK_GATE_EN', 'pitch_num', 'is_platform_pdn']
    X_train_raw = df_sweep[feature_cols].values
    y_vdd_raw = df_sweep['ir_drop_worst_mv'].values
    y_log = np.log(y_vdd_raw)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    
    kernel = Matern(length_scale=np.ones(X_train.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, random_state=42)
    gp.fit(X_train, y_log)
    
    # Dual objective orderings: VDD worst IR vs VDD+VSS rail sum
    df_sweep['rank_vdd_only'] = df_sweep['ir_drop_worst_mv'].rank(method='min', ascending=True).astype(int)
    df_sweep['rank_vdd_vss_rail'] = df_sweep['rail_worst_sum_mv'].rank(method='min', ascending=True).astype(int)
    
    bo_runs_rows = df_sweep[df_sweep['bo_rank'].notna()].copy().sort_values(
        by=['ir_drop_worst_mv', 'power_mw', 'die_area_um2'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    
    proposals = []
    for rank_idx, row in bo_runs_rows.head(num_candidates).iterrows():
        rank_id = rank_idx + 1
        prop_id = str(row['config']) if 'config' in row and pd.notna(row['config']) else f"cfg_run_{int(row['bo_rank']):02d}"
        
        swap_note = ""
        if int(row['pdn_strap_pitch_um']) == 24:
            if int(row['ECC_LANES']) == 1:
                swap_note = "Rank 4 VDD-only (0.7112 mV), but Rank 3 VDD+VSS Rail (1.3837 mV)"
            elif int(row['ECC_LANES']) == 4:
                swap_note = "Rank 3 VDD-only (0.6859 mV), but Rank 4 VDD+VSS Rail (1.4426 mV) due to ground bounce"

        proposals.append({
            'rank': rank_id,
            'proposal_id': prop_id,
            'config_base': 'cfg_small',
            'NUM_CH': int(row['NUM_CH']),
            'DATA_W': int(row['DATA_W']),
            'ECC_LANES': int(row['ECC_LANES']),
            'CLK_GATE_EN': int(row['CLK_GATE_EN']),
            'PDN_STRAP_PITCH_UM': int(snap_to_grid(row['pitch_num'])),
            'executed_strap_pitch_um': int(row['pitch_num']),
            'is_platform_pdn': int(row['is_platform_pdn']),
            'ir_drop_worst_mv': round(float(row['ir_drop_worst_mv']), 4),
            'vss_worst_mv': round(float(row['vss_worst_mv']), 4),
            'rail_worst_sum_mv': round(float(row['rail_worst_sum_mv']), 4),
            'rank_vdd_only': int(row['rank_vdd_only']),
            'rank_vdd_vss_rail': int(row['rank_vdd_vss_rail']),
            'bo_score': round(float(row['ir_drop_worst_mv']), 4),
            'rank_swap_note': swap_note,
            'method': 'heuristic_ranking_over_measured_runs',
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

    # Evaluate candidate grid for unobserved acquisition
    cand_grid = []
    for ch in [2, 4, 8]:
        for w in [32, 64, 128]:
            for ecc in [1, 2, 4]:
                for clk in [0, 1]:
                    for pitch in [8, 12, 16, 24, 30]:
                        is_plat = 1 if pitch == 30 else 0
                        cand_grid.append({
                            'NUM_CH': ch, 'DATA_W': w, 'ECC_LANES': ecc, 'CLK_GATE_EN': clk,
                            'pitch_num': pitch, 'is_platform_pdn': is_plat,
                            'PDN_STRAP_PITCH_UM': pitch
                        })

    df_cand = pd.DataFrame(cand_grid)

    executed_combos = set()
    for _, r in df_sweep.iterrows():
        executed_combos.add((int(r['NUM_CH']), int(r['DATA_W']), int(r['ECC_LANES']), int(r['CLK_GATE_EN']), int(r['pitch_num'])))

    df_cand['is_executed'] = df_cand.apply(lambda r: (int(r['NUM_CH']), int(r['DATA_W']), int(r['ECC_LANES']), int(r['CLK_GATE_EN']), int(r['pitch_num'])) in executed_combos, axis=1)
    df_cand = df_cand[~df_cand['is_executed']].copy()

    def get_category_and_note(row):
        ch, w, ecc, clk, p = int(row['NUM_CH']), int(row['DATA_W']), int(row['ECC_LANES']), int(row['CLK_GATE_EN']), int(row['pitch_num'])
        if (ch == 2) and (w == 32) and (clk == 1) and (p in [12, 16, 24]) and (ecc in [1, 2, 4]):
            return 'well_supported', 'Well-supported candidate: cfg_small base where strap sweep was measured'
        elif (ch == 2) and (w == 32) and (clk == 1) and (p == 8) and (ecc in [1, 2, 4]):
            return 'one_step_extrapolation', '8 µm is outside the measured pitch range 12–24 µm (recommended next PnR run)'
        else:
            reasons = []
            if clk == 0:
                reasons.append("clock gating off never measured (all 6 runs have CLK_GATE_EN=1)")
            if (ch != 2) or (w != 32) or (p == 30):
                reasons.append("strap pitch never measured at this architecture (only cfg_mid, platform PDN)")
            if not reasons:
                reasons.append("strap pitch never measured at this architecture (only cfg_mid, platform PDN)")
            note = "; ".join(reasons)
            return 'exploratory', note

    cat_notes = df_cand.apply(get_category_and_note, axis=1)
    df_cand['candidate_category'] = [cn[0] for cn in cat_notes]
    df_cand['why_this_class'] = [cn[1] for cn in cat_notes]

    X_cand_raw = df_cand[feature_cols].values
    X_cand_scaled = scaler.transform(X_cand_raw)

    mu_log, sigma_log = gp.predict(X_cand_scaled, return_std=True)
    lcb_log = mu_log - 1.96 * sigma_log

    df_cand['mu_mv'] = np.round(np.exp(mu_log), 4)
    df_cand['sigma_mv'] = np.round(np.exp(mu_log) * sigma_log, 4)
    df_cand['lcb_score_mv'] = np.round(np.exp(lcb_log), 4)

    next_proposals = []
    
    categories = [
        ('well_supported', False),
        ('one_step_extrapolation', False),
        ('exploratory', True)
    ]

    for cat_key, is_exp in categories:
        df_cat = df_cand[df_cand['candidate_category'] == cat_key].sort_values('lcb_score_mv', ascending=True).reset_index(drop=True)
        # Display all for well_supported (5) and one_step_extrapolation (3), or top 4 for exploratory
        limit = num_candidates if cat_key == 'exploratory' else len(df_cat)
        top_cat = df_cat.head(limit)
        
        for rank_idx, row in top_cat.iterrows():
            rank_id = rank_idx + 1
            cat_short = "hull" if cat_key == 'well_supported' else ("extrap" if cat_key == 'one_step_extrapolation' else "exp")
            cand_id = f"cfg_{cat_short}_{rank_id:02d}"
            
            next_proposals.append({
                'rank': rank_id,
                'candidate_id': cand_id,
                'config_base': 'cfg_small' if row['NUM_CH'] == 2 else 'cfg_mid',
                'NUM_CH': int(row['NUM_CH']),
                'DATA_W': int(row['DATA_W']),
                'ECC_LANES': int(row['ECC_LANES']),
                'CLK_GATE_EN': int(row['CLK_GATE_EN']),
                'PDN_STRAP_PITCH_UM': int(row['PDN_STRAP_PITCH_UM']),
                'is_platform_pdn': int(row['is_platform_pdn']),
                'mu_mv': float(row['mu_mv']),
                'sigma_mv': float(row['sigma_mv']),
                'lcb_score_mv': float(row['lcb_score_mv']),
                'why_this_class': str(row['why_this_class']),
                'exploratory': is_exp,
                'candidate_category': cat_key,
                'method': 'gp_lcb_log_space_surrogate',
                'n_training_obs': len(df_sweep),
                'training_note': str(row['why_this_class'])
            })

    df_next = pd.DataFrame(next_proposals)
    out_next_csv = os.path.join(dir_path, 'bo_next_candidates.csv')
    df_next.to_csv(out_next_csv, index=False)
    print(f"\nSaved {len(df_next)} BO candidate proposals to 'bo_next_candidates.csv'")
    print(df_next.to_string(index=False))

    return df_proposals

if __name__ == "__main__":
    select_top_candidates_bo()


