import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def compute_model_metrics(csv_path='predictions.csv', out_csv='model_metrics.csv'):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(dir_path, csv_path)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{csv_path}' not found at {input_file}")

    df = pd.read_csv(input_file, comment='#')

    results = []
    
    partitions = ['holdout', 'train', 'calib', 'all']
    models = [
        ('Hybrid Physics+ML Residual', 'pred_v'),
        ('Physics-Only Baseline', 'coarse_v')
    ]

    for part in partitions:
        sub = df if part == 'all' else df[df['partition'] == part]
        if sub.empty:
            continue
            
        for model_label, mcol in models:
            label_v = sub['label_v'].values
            pred_v = sub[mcol].values
            
            mae_mv = float(np.mean(np.abs(pred_v - label_v))) * 1000.0
            rmse_mv = float(np.sqrt(np.mean((pred_v - label_v) ** 2))) * 1000.0
            r2 = float(r2_score(label_v, pred_v))
            
            mape = float(np.mean(np.abs(label_v - pred_v) / label_v)) * 100.0
            accuracy = 100.0 - mape
            
            # Top-5% Spatial Hit Rate (k=29 tiles per design x scenario)
            hits = []
            for _, g in sub.groupby(['design', 'scenario']):
                top_label = set(g.nlargest(29, 'label_v')['tile_id'])
                top_pred = set(g.nlargest(29, mcol)['tile_id'])
                hits.append(len(top_label.intersection(top_pred)) / 29.0)
            top5_hit = float(np.mean(hits)) * 100.0 if hits else 0.0
            
            conformal_cov = float(np.mean((sub['label_v'] >= sub['lo_v']) & (sub['label_v'] <= sub['hi_v']))) * 100.0
            
            results.append({
                'partition': part,
                'model': model_label,
                'model_col': mcol,
                'mae_mv': round(mae_mv, 4),
                'rmse_mv': round(rmse_mv, 4),
                'r2_score': round(r2, 4),
                'mape_pct': round(mape, 2),
                'accuracy_pct': round(accuracy, 2),
                'top5_hit_rate_pct': round(top5_hit, 2),
                'conformal_coverage_pct': round(conformal_cov, 2)
            })

    df_res = pd.DataFrame(results)
    out_file = os.path.join(dir_path, out_csv)
    df_res.to_csv(out_file, index=False)
    print(f"Saved model metrics to '{out_csv}' ({len(df_res)} rows).")
    print(df_res.to_string(index=False))
    return df_res

if __name__ == "__main__":
    compute_model_metrics()
