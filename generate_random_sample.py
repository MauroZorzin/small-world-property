#!/usr/bin/env python3
"""
Generates random directed graphs preserving the degree sequence of a real
graph read from degree_sequence.json (never re-parses the .dot file)
Computes their Fagiolo and path-length metrics and appends one row per graph
to its OWN output CSV
One file per invocation so many instances can run in parallel with no
shared-file write contention
Each graph is discarded immediately after its metrics are computed
and the row is flushed to disk

Two modes, mutually exclusive
    --count N                  generate exactly N random graphs
    --converge-threshold PCT   generate in batches of --batch-size, checking
                                after each batch whether every tracked metric
                                is under PCT relative SEM across ALL accumulated
                                samples (this run plus every prior random_samples
                                file); stops once converged or at --max-samples

Usage
    python generate_random_sample.py results/<project>/degree_sequence.json \\
        --out-dir results/<project>/ --count 50 [--seed 42]
    python generate_random_sample.py results/<project>/degree_sequence.json \\
        --out-dir results/<project>/ --converge-threshold 2.0
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import metrics_logic as ml
import random_graph as rg


METRIC_COLUMNS = (
    [f'clustering_full_{k}'   for k in ml.C_KEYS]
    + [f'clustering_lscc_{k}'   for k in ml.C_KEYS]
    + [f'clustering_allscc_{k}' for k in ml.C_KEYS]
    + [f'path_length_{k}' for k in ml.L_KEYS]
)
FIELDNAMES = (
    ['run_id', 'sample_index', 'timestamp',
     'num_nodes', 'num_edges',
     'degree_sequence_matches_original', 'max_in_degree', 'max_out_degree',
     'n_self_loops_repaired', 'n_multiedges_repaired', 'n_dropped_edges']
    + METRIC_COLUMNS
)


def generate_one_sample(rng, in_seq, out_seq, target_in_sorted, target_out_sorted, run_id, sample_index):
    G_rand, n_self, n_multi, n_dropped = rg.generate_random_directed_graph(in_seq, out_seq, rng)

    actual_in_sorted  = sorted(d for _, d in G_rand.in_degree())
    actual_out_sorted = sorted(d for _, d in G_rand.out_degree())
    matches = (actual_in_sorted == target_in_sorted and actual_out_sorted == target_out_sorted)

    clustering   = ml.fagiolo_scoped(G_rand)
    path_lengths = ml.all_path_lengths(G_rand)

    row = {
        'run_id': run_id,
        'sample_index': sample_index,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'num_nodes': G_rand.number_of_nodes(),
        'num_edges': G_rand.number_of_edges(),
        'degree_sequence_matches_original': matches,
        'max_in_degree': max(actual_in_sorted) if actual_in_sorted else 0,
        'max_out_degree': max(actual_out_sorted) if actual_out_sorted else 0,
        'n_self_loops_repaired': n_self,
        'n_multiedges_repaired': n_multi,
        'n_dropped_edges': n_dropped,
    }
    for scope in ('full', 'lscc', 'allscc'):
        for k in ml.C_KEYS:
            row[f'clustering_{scope}_{k}'] = clustering[scope][k]
    for k in ml.L_KEYS:
        row[f'path_length_{k}'] = path_lengths[k]
    return row, matches


def accumulated_sample_count(samples_dir):
    total = 0
    for p in samples_dir.glob('*.csv'):
        try:
            with open(p, encoding='utf-8') as fh:
                total += max(0, sum(1 for _ in fh) - 1)
        except Exception:
            continue
    return total


def check_convergence(samples_dir, threshold_pct):
    """Worst (metric, relative_sem_pct) across every accumulated sample file, and whether all are under threshold"""
    frames = []
    for p in samples_dir.glob('*.csv'):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        return False, None, float('inf')

    df = pd.concat(frames, ignore_index=True)
    worst_metric, worst_pct = None, -1.0
    for col in METRIC_COLUMNS:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna() if col in df else pd.Series(dtype=float)
        pct = ml.relative_sem_pct(vals.mean(), vals.std(), len(vals)) if len(vals) else float('nan')
        if not np.isfinite(pct):
            pct = float('inf')  # can't confirm convergence yet, force another batch
        if pct > worst_pct:
            worst_metric, worst_pct = col, pct
    return worst_pct <= threshold_pct, worst_metric, worst_pct


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('degree_sequence_json', type=str,
                         help='Path to degree_sequence.json produced by compute_real_metrics.py')
    parser.add_argument('--out-dir', type=str, required=True,
                         help='Project directory (a random_samples/ subfolder is created here)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--count', type=int, default=None, help='Fixed number of random graphs to generate')
    mode.add_argument('--converge-threshold', type=float, default=None,
                       help='Generate in batches until every tracked metric is under this relative SEM percent')
    parser.add_argument('--batch-size', type=int, default=50,
                         help='Samples per batch before rechecking convergence (converge mode only, default 50)')
    parser.add_argument('--max-samples', type=int, default=5000,
                         help='Safety cap on total accumulated samples in converge mode (default 5000)')
    parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed for reproducibility')
    args = parser.parse_args()

    if args.count is None and args.converge_threshold is None:
        args.count = 1

    deg_path = Path(args.degree_sequence_json)
    if not deg_path.exists():
        print(f"Error '{deg_path}' not found run compute_real_metrics.py first", file=sys.stderr)
        sys.exit(1)

    with open(deg_path, encoding='utf-8') as f:
        deg = json.load(f)
    in_seq, out_seq = deg['in_seq'], deg['out_seq']
    target_in_sorted  = sorted(in_seq)
    target_out_sorted = sorted(out_seq)

    samples_dir = Path(args.out_dir) / 'random_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    out_path = samples_dir / f'run_{run_id}.csv'

    rng = random.Random(args.seed)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        def write_batch(sample_index, count):
            for i in range(count):
                row, matches = generate_one_sample(rng, in_seq, out_seq, target_in_sorted, target_out_sorted,
                                                     run_id, sample_index + i)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                print(f"  [{sample_index + i + 1}] sample written (degree_ok={matches})", file=sys.stderr)
            return sample_index + count

        if args.converge_threshold is not None:
            sample_index = 0
            total = accumulated_sample_count(samples_dir)
            while True:
                batch = min(args.batch_size, args.max_samples - total)
                if batch <= 0:
                    print(f"Reached --max-samples cap ({args.max_samples}) before full convergence", file=sys.stderr)
                    break
                sample_index = write_batch(sample_index, batch)
                total += batch
                converged, worst_metric, worst_pct = check_convergence(samples_dir, args.converge_threshold)
                print(f"  {total} accumulated samples | worst metric {worst_metric} at {worst_pct:.2f}% "
                      f"(threshold {args.converge_threshold}%)", file=sys.stderr)
                if converged:
                    print(f"Converged: every tracked metric is under {args.converge_threshold}% relative SEM",
                          file=sys.stderr)
                    break
        else:
            write_batch(0, args.count)

    print(f"Saved samples to: {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
