#!/usr/bin/env python3
"""
Reads real_metrics.csv + all random_samples/*.csv fragments for a project,
computes the 45 small-world sigma values, and writes final_summary.csv.
Re-runnable at any time as more random-sample fragments accumulate — nothing
already computed is ever recomputed.

Usage:
    python combine_results.py --project-dir results/<project>/
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import metrics_logic as ml


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-dir', type=str, required=True)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    real_metrics_path = project_dir / 'real_metrics.csv'
    if not real_metrics_path.exists():
        print(f"Error: {real_metrics_path} not found — run compute_real_metrics.py first",
              file=sys.stderr)
        sys.exit(1)

    real = pd.read_csv(real_metrics_path).iloc[0].to_dict()

    fragment_paths = sorted(glob.glob(str(project_dir / 'random_samples' / '*.csv')))
    if not fragment_paths:
        print(f"Error: no random-sample files found under {project_dir / 'random_samples'} "
              f"— run generate_random_sample.py first", file=sys.stderr)
        sys.exit(1)

    samples = pd.concat([pd.read_csv(p) for p in fragment_paths], ignore_index=True)
    print(f"Loaded {len(samples)} random samples from {len(fragment_paths)} file(s)", file=sys.stderr)

    n_bad = int((~samples['degree_sequence_matches_original']).sum())
    if n_bad:
        print(f"Warning: {n_bad}/{len(samples)} random samples have a degree-sequence "
              f"mismatch (residual drift from unrepairable self-loops/multi-edges)",
              file=sys.stderr)

    results = dict(real)

    C_orig_all = {
        scope: {k: real[f'clustering_{scope}_{k}'] for k in ml.C_KEYS}
        for scope in ml.C_SCOPES
    }
    L_orig = {k: real[f'path_length_{k}'] for k in ml.L_KEYS}

    C_rand_mean, C_rand_std = {}, {}
    for scope in ml.C_SCOPES:
        C_rand_mean[scope] = {}
        C_rand_std[scope] = {}
        for k in ml.C_KEYS:
            col = samples[f'clustering_{scope}_{k}']
            col = col[np.isfinite(col)]
            C_rand_mean[scope][k] = float(col.mean()) if len(col) else float('nan')
            C_rand_std[scope][k]  = float(col.std())  if len(col) > 1 else 0.0

    L_rand_mean, L_rand_std = {}, {}
    for k in ml.L_KEYS:
        col = samples[f'path_length_{k}']
        col = col[np.isfinite(col)]
        L_rand_mean[k] = float(col.mean()) if len(col) else float('nan')
        L_rand_std[k]  = float(col.std())  if len(col) > 1 else 0.0

    def _sigma(scope, c_key, l_key):
        C, L = C_orig_all[scope][c_key], L_orig[l_key]
        Cr, Lr = C_rand_mean[scope][c_key], L_rand_mean[l_key]
        if all(np.isfinite(v) for v in [C, L, Cr, Lr]) and Cr > 0 and Lr > 0:
            return float((C / Cr) / (L / Lr))
        return float('nan')

    for scope in ml.C_SCOPES:
        for k in ml.C_KEYS:
            results[f'smallworld_C_orig_{scope}_{k}']      = C_orig_all[scope][k]
            results[f'smallworld_C_rand_mean_{scope}_{k}'] = C_rand_mean[scope][k]
            results[f'smallworld_C_rand_std_{scope}_{k}']  = C_rand_std[scope][k]

    for k in ml.L_KEYS:
        results[f'smallworld_L_orig_{k}']      = L_orig[k]
        results[f'smallworld_L_rand_mean_{k}'] = L_rand_mean[k]
        results[f'smallworld_L_rand_std_{k}']  = L_rand_std[k]

    for c_key in ml.C_KEYS:
        for scope in ml.C_SCOPES:
            for l_key in ml.L_KEYS:
                s = _sigma(scope, c_key, l_key)
                results[f'smallworld_sigma_{c_key}_{scope}_{l_key}']         = s
                results[f'smallworld_is_smallworld_{c_key}_{scope}_{l_key}'] = (
                    bool(s > 1) if np.isfinite(s) else False
                )

    results['smallworld_num_random_samples'] = len(samples)

    out_path = project_dir / 'final_summary.csv'
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(samples)} random samples used)", file=sys.stderr)


if __name__ == '__main__':
    main()
