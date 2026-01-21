#!/usr/bin/env python3
"""
aggregate_pipeline.py
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ----------------- Config -----------------
OUTPUT_DIR = 'aggregate_output'
PER_PROJECT_DIR = os.path.join(OUTPUT_DIR, 'per_project_correlations')
os.makedirs(PER_PROJECT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DV8_REPORTS_DIR_NAME = 'dv8-reports'
RESULTS_DIR_NAME = 'results'

ANTI_PATTERN_CSV_PATTERNS = ['*clique*.csv', '*package*.csv', '*unhealthy*.csv', '*crossing*.csv', '*.csv']

QUALITY_METRICS = [
    'total_anti_patterns',
    'quality_score',
    'numUnhealthyInheritance',
    'numClique',
    'numPackageCycle'
]

SW_METRICS = [
    'clustering_overall',
    'clustering_cycle',
    'clustering_middleman',
    'total_degree',
    'in_degree',
    'out_degree',
    'bilateral_degree'
]

# ----------------- Helpers -----------------
def normalize_path_str(s):
    if pd.isna(s):
        return ''
    return str(s).strip().strip('"').strip("'").replace('\\', '/')

def find_projects(root_dir):
    projects = {}
    for base, key in [(DV8_REPORTS_DIR_NAME, 'dv8_analysis'), (RESULTS_DIR_NAME, 'results_path')]:
        root = os.path.join(root_dir, base)
        if os.path.isdir(root):
            for p in os.listdir(root):
                d = os.path.join(root, p)
                if base == DV8_REPORTS_DIR_NAME:
                    d = os.path.join(d, 'dv8-analysis-result')
                if os.path.isdir(d):
                    projects.setdefault(p, {})[key] = d
    return projects

def load_file_measure(d):
    p = os.path.join(d, 'file-measure-report.csv')
    return pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()

def load_node_and_network_results(d):
    n = os.path.join(d, 'results-per-node.csv')
    net = os.path.join(d, 'results.csv')
    return (
        pd.read_csv(n, low_memory=False) if os.path.exists(n) else pd.DataFrame(),
        pd.read_csv(net, low_memory=False) if os.path.exists(net) else pd.DataFrame()
    )

def load_anti_patterns(d):
    base = os.path.join(d, 'anti-pattern', 'anti-pattern-costs')
    if not os.path.isdir(base):
        base = os.path.join(d, 'anti-pattern')
    frames = []
    for pat in ANTI_PATTERN_CSV_PATTERNS:
        for f in glob.glob(os.path.join(base, pat)):
            try:
                frames.append(pd.read_csv(f, low_memory=False))
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def pick_node_column(df):
    for c in ['node_name', 'name', 'Name']:
        if c in df.columns:
            return c
    return None

def prepare_project_merge(fm, nr):
    if 'FileName' not in fm.columns:
        for alt in ['File', 'filename', 'file']:
            if alt in fm.columns:
                fm = fm.rename(columns={alt: 'FileName'})

    node_col = pick_node_column(nr)
    if not node_col:
        raise KeyError("No node column found")

    fm['FileName'] = fm['FileName'].map(normalize_path_str)
    nr[node_col] = nr[node_col].map(normalize_path_str)

    merged = pd.merge(fm, nr, left_on='FileName', right_on=node_col, how='inner')

    anti_cols = [
        'numClique', 'numCrossing', 'numModularityViolation',
        'numPackageCycle', 'numUnhealthyInheritance', 'numUnstableInterface'
    ]
    existing = [c for c in anti_cols if c in merged.columns]
    merged['total_anti_patterns'] = merged[existing].sum(axis=1) if existing else 0
    merged['quality_score'] = 1 / (1 + merged['total_anti_patterns'])
    return merged

def compute_project_correlations(df, project):
    rows = []
    for q in QUALITY_METRICS:
        for s in SW_METRICS:
            if q in df.columns and s in df.columns:
                mask = df[[q, s]].notna().all(axis=1)
                if mask.sum() >= 4:
                    pr, pp = pearsonr(df.loc[mask, q], df.loc[mask, s])
                    sr, sp = spearmanr(df.loc[mask, q], df.loc[mask, s])
                    rows.append({
                        'Project': project,
                        'Quality Metric': q,
                        'Small-World Metric': s,
                        'N': mask.sum(),
                        'Pearson r': pr,
                        'Pearson p-value': pp,
                        'Spearman r': sr,
                        'Spearman p-value': sp
                    })
    return pd.DataFrame(rows)

# ----------------- Pipeline -----------------
def run_aggregation(root_dir='.'):
    projects = find_projects(root_dir)
    all_corrs = []

    plot_raw = []

    for proj, info in projects.items():
        if 'dv8_analysis' not in info or 'results_path' not in info:
            continue

        merged = prepare_project_merge(
            load_file_measure(info['dv8_analysis']),
            load_node_and_network_results(info['results_path'])[0]
        )

        if merged.empty:
            continue

        size = merged['FileName'].nunique()
        mean_sw = merged['clustering_overall'].mean()
        total_ap = merged['total_anti_patterns'].sum()

        plot_raw.append((proj, mean_sw, total_ap, size))

        corr = compute_project_correlations(merged, proj)
        if not corr.empty:
            corr.to_csv(os.path.join(PER_PROJECT_DIR, f'{proj}_correlations.csv'), index=False)
            all_corrs.append(corr)

    aggregated = pd.concat(all_corrs, ignore_index=True)
    aggregated.to_csv(os.path.join(OUTPUT_DIR, 'aggregated_correlations.csv'), index=False)

    df_raw = pd.DataFrame(plot_raw, columns=['Project','SW','Issues','Size'])

    # ----------------- Scatter plots -----------------
    sns.regplot(data=df_raw, x='SW', y='Issues', ci=95)
    plt.title('SW vs Architectural Issues (Raw)')
    plt.savefig(os.path.join(OUTPUT_DIR,'sw_vs_arch_issues_raw.png'), dpi=200)
    plt.close()

    # ----------------- Partial correlation -----------------
    X = df_raw[['SW','Size']]
    y = df_raw['Issues']

    lr_x = LinearRegression().fit(X[['Size']], X['SW'])
    lr_y = LinearRegression().fit(X[['Size']], y)

    sw_res = X['SW'] - lr_x.predict(X[['Size']])
    y_res = y - lr_y.predict(X[['Size']])

    sns.regplot(x=sw_res, y=y_res, ci=95)
    plt.title('Partial Correlation: SW vs Issues | controlling for Size')
    plt.savefig(os.path.join(OUTPUT_DIR,'sw_vs_arch_issues_partial_corr.png'), dpi=200)
    plt.close()

    # ----------------- Log–log plot -----------------
    df_log = df_raw[(df_raw['SW'] > 0) & (df_raw['Issues'] > 0)].copy()
    df_log['logSW'] = np.log10(df_log['SW'])
    df_log['logIssues'] = np.log10(df_log['Issues'])

    sns.regplot(data=df_log, x='logSW', y='logIssues', ci=95)
    plt.title('Log–Log: SW vs Architectural Issues')
    plt.savefig(os.path.join(OUTPUT_DIR,'sw_vs_arch_issues_loglog.png'), dpi=200)
    plt.close()

    analyze_aggregated(aggregated)

# ----------------- Aggregated analysis -----------------
def analyze_aggregated(df):
    # Boxplot
    sns.boxplot(data=df, x='Small-World Metric', y='Pearson r')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'pearson_r_boxplot_by_sw_metric.png'), dpi=200)
    plt.close()

    # -------- HEATMAP (RESTORED) --------
    pivot = df.pivot_table(
        index='Quality Metric',
        columns='Small-World Metric',
        values='Pearson r',
        aggfunc='mean'
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0
    )
    plt.title('Mean Pearson r: Quality vs Small-World Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'pearson_r_heatmap.png'), dpi=200)
    plt.close()

# ----------------- Main -----------------
if __name__ == '__main__':
    root = '.' if len(sys.argv) < 2 else sys.argv[1]
    run_aggregation(root)
    print("Done. Outputs under:", os.path.abspath(OUTPUT_DIR))
