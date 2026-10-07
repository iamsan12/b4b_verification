import pandas as pd
import numpy as np
import os
import sys

from prism_risk_engine import NX_COARSE, NY_COARSE

# Physical sensor placement parameters
NUM_SENSORS = 4            # Silicon area/cost budget constraint: 4 TDET/PDET sensor macro cells
SENSOR_RADIUS = 2.5        # Effective sensing radius in grid units (~46.3 µm physical sensing range)
KERNEL_TYPE = "exponential sensing kernel, e-folding 2.5 tiles (46 µm), cut-off at 10%"


def load_grid_risk_map(risk_csv='prism_tile_attribution.csv', inst_csv='instances.csv', stats_csv='design_stats.csv', nx=NX_COARSE, ny=NY_COARSE):
    """
    Loads tile risk attribution and sums instance risks into a 24x24 coarse grid
    using REAL physical silicon coordinates (x_um, y_um).
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    attr_path = os.path.join(dir_path, risk_csv) if not os.path.isabs(risk_csv) else risk_csv
    inst_path = os.path.join(dir_path, inst_csv) if not os.path.isabs(inst_csv) else inst_csv
    stats_path = os.path.join(dir_path, stats_csv) if not os.path.isabs(stats_csv) else stats_csv

    if not os.path.exists(attr_path):
        raise FileNotFoundError(f"Required telemetry input file missing: '{attr_path}'")
    if not os.path.exists(inst_path):
        raise FileNotFoundError(f"Required telemetry input file missing: '{inst_path}'")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Required telemetry input file missing: '{stats_path}'")

    st = pd.read_csv(stats_path)
    die_w = float(st['die_w_um'].iloc[0])
    die_h = float(st['die_h_um'].iloc[0])

    print(f"[TELEMETRY] Loading real silicon placement coordinates from '{os.path.basename(inst_path)}'...")
    df_attr = pd.read_csv(attr_path)
    df_inst = pd.read_csv(inst_path)

    merged = df_attr.merge(df_inst[['inst_id', 'x_um', 'y_um']], left_on='inst_idx', right_on='inst_id', how='left')

    risk_grid = np.zeros((ny, nx))

    for _, row in merged.iterrows():
        if pd.isna(row['x_um']) or pd.isna(row['y_um']):
            raise KeyError(f"inst_id {row['inst_idx']} missing coordinates in instances.csv")
        x, y = float(row['x_um']), float(row['y_um'])
        risk_ps = float(row['total_delay_penalty_ps'])
        
        gx = min(nx - 1, max(0, int(x / (die_w / nx))))
        gy = min(ny - 1, max(0, int(y / (die_h / ny))))
        risk_grid[gy, gx] += risk_ps

    grid_df = pd.DataFrame(risk_grid)
    grid_df.to_csv(os.path.join(dir_path, 'prism_chip_risk_grid.csv'), index=False)

    return risk_grid, die_w, die_h


def sensor_coverage_kernel(pos, grid_shape, radius=SENSOR_RADIUS):
    """Exponential decay coverage kernel: Coverage weight C(x,y) = exp(-dist / radius)"""
    gh, gw = grid_shape
    ys, xs = pos
    y_coords, x_coords = np.ogrid[:gh, :gw]
    distances = np.sqrt((x_coords - xs)**2 + (y_coords - ys)**2)
    coverage = np.exp(-distances / radius)
    coverage[coverage < 0.1] = 0.0
    return coverage


def run_greedy_sensor_placement(risk_grid, die_w=444.22, die_h=444.22, num_sensors=NUM_SENSORS, sensor_radius=SENSOR_RADIUS, out_filename='telemetry_sensor_placements.csv'):
    """Greedy Submodular Maximum-Coverage Algorithm using real physical silicon floorplan coordinates."""
    gh, gw = risk_grid.shape
    total_grid_risk = np.sum(risk_grid)
    if total_grid_risk <= 0:
        total_grid_risk = 1e-6

    placed_sensors = []
    current_coverage = np.zeros((gh, gw))
    covered_risk_history = []

    print(f"\n[TELEMETRY PLACEMENT] Running Greedy Submodular Optimization for K={num_sensors} sensors...")
    print(f"Total Unmitigated Chip Risk: {total_grid_risk:.2f} ps\n")

    for k in range(num_sensors):
        best_candidate = None
        best_marginal_gain = -1.0
        best_sensor_coverage = None

        for y in range(gh):
            for x in range(gw):
                if (y, x) in placed_sensors:
                    continue

                cov_kernel = sensor_coverage_kernel((y, x), (gh, gw), radius=sensor_radius)
                new_coverage = np.maximum(current_coverage, cov_kernel)
                marginal_gain = np.sum((new_coverage - current_coverage) * risk_grid)

                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_candidate = (y, x)
                    best_sensor_coverage = cov_kernel

        placed_sensors.append(best_candidate)
        current_coverage = np.maximum(current_coverage, best_sensor_coverage)
        total_covered = np.sum(current_coverage * risk_grid)
        covered_risk_history.append(best_marginal_gain)

        x_um = round((best_candidate[1] + 0.5) * (die_w / gw), 2)
        y_um = round((best_candidate[0] + 0.5) * (die_h / gh), 2)

        print(f" Sensor {k+1}: Grid ({best_candidate[1]}, {best_candidate[0]}) -> Physical Silicon (x={x_um} um, y={y_um} um) | "
              f"Marginal Risk Covered: +{best_marginal_gain:.2f} ps | "
              f"Total Covered: {total_covered:.2f} ps ({(total_covered/total_grid_risk)*100:.1f}%)")

    total_covered_risk = np.sum(current_coverage * risk_grid)
    coverage_pct = (total_covered_risk / total_grid_risk) * 100.0
    submodular_bound_pct = (1.0 - 1.0 / np.e) * 100.0

    print("\n------------------------------------------------------------")
    print(" GREEDY SUBMODULAR APPROXIMATION GUARANTEE ")
    print("------------------------------------------------------------")
    print(f" Total Sensors Placed        : {num_sensors}")
    print(f" Total Risk Monitored       : {total_covered_risk:.2f} / {total_grid_risk:.2f} ps")
    print(f" Measured Risk Coverage     : {coverage_pct:.1f}%")
    print(f" Theoretical Guarantee      : (1 - 1/e) ~ {submodular_bound_pct:.1f}% approximation")
    print("------------------------------------------------------------")

    df_sensors = pd.DataFrame([
        {
            'sensor_id': f"TDET_PDET_S{idx+1:02d}",
            'real_x_um': round((loc[1] + 0.5) * (die_w / gw), 2),
            'real_y_um': round((loc[0] + 0.5) * (die_h / gh), 2),
            'grid_x': loc[1],
            'grid_y': loc[0],
            'marginal_risk_covered_ps': round(gain, 2),
            'num_sensors': num_sensors,
            'sensor_radius_grid_units': sensor_radius,
            'kernel_type': KERNEL_TYPE
        }
        for idx, (loc, gain) in enumerate(zip(placed_sensors, covered_risk_history))
    ])

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(out_dir, out_filename)
    df_sensors.to_csv(out_csv, index=False)
    print(f"Saved sensor placement coordinates to '{out_csv}'\n")

    return df_sensors


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    risk_grid, die_w, die_h = load_grid_risk_map()
    df_sensors = run_greedy_sensor_placement(risk_grid, die_w, die_h, num_sensors=NUM_SENSORS, sensor_radius=SENSOR_RADIUS)

    # Precompute per-scenario sensor placements for all supported droop sources
    droop_sources = ['measured_rail', 'measured_vdd', 'measured_cell_rail', 'measured_p12_rail', 'guardband_5pct', 'guardband_10pct', 'roleB_syn_corpus_whatif']
    for sc in droop_sources:
        grid_f = os.path.join(dir_path, f'prism_chip_risk_grid_{sc}.csv')
        if os.path.exists(grid_f):
            grid_sc = pd.read_csv(grid_f).values
            run_greedy_sensor_placement(grid_sc, die_w, die_h, num_sensors=NUM_SENSORS, sensor_radius=SENSOR_RADIUS, out_filename=f'telemetry_sensor_placements_{sc}.csv')
