#!/usr/bin/env python3
"""
Generates N random directed graphs preserving the degree sequence of a real
graph (read from degree_sequence.json — never re-parses the .dot file),
computes their Fagiolo/path-length metrics, and appends one row per graph to
its OWN output CSV: one file per invocation, so many instances can run in
parallel with no shared-file write contention. Each graph is discarded
immediately after its metrics are computed and the row is flushed to disk.

Usage:
    python generate_random_sample.py results/<project>/degree_sequence.json \\
        --out-dir results/<project>/ --count 50 [--seed 42]
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import metrics_logic as ml
import random_graph as rg


FIELDNAMES = (
    ['run_id', 'sample_index', 'timestamp',
     'num_nodes', 'num_edges',
     'degree_sequence_matches_original', 'max_in_degree', 'max_out_degree',
     'n_self_loops_repaired', 'n_multiedges_repaired', 'n_dropped_edges']
    + [f'clustering_full_{k}'   for k in ml.C_KEYS]
    + [f'clustering_lscc_{k}'   for k in ml.C_KEYS]
    + [f'clustering_allscc_{k}' for k in ml.C_KEYS]
    + [f'path_length_{k}' for k in ml.L_KEYS]
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('degree_sequence_json', type=str,
                         help='Path to degree_sequence.json produced by compute_real_metrics.py')
    parser.add_argument('--out-dir', type=str, required=True,
                         help='Project directory (a random_samples/ subfolder is created here)')
    parser.add_argument('--count', type=int, default=1, help='Number of random graphs to generate')
    parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed for reproducibility')
    args = parser.parse_args()

    deg_path = Path(args.degree_sequence_json)
    if not deg_path.exists():
        print(f"Error: '{deg_path}' not found — run compute_real_metrics.py first", file=sys.stderr)
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

        for i in range(args.count):
            G_rand, n_self, n_multi, n_dropped = rg.generate_random_directed_graph(
                in_seq, out_seq, rng
            )

            actual_in_sorted  = sorted(d for _, d in G_rand.in_degree())
            actual_out_sorted = sorted(d for _, d in G_rand.out_degree())
            matches = (actual_in_sorted == target_in_sorted and
                       actual_out_sorted == target_out_sorted)

            clustering   = ml.fagiolo_scoped(G_rand)
            path_lengths = ml.all_path_lengths(G_rand)

            row = {
                'run_id': run_id,
                'sample_index': i,
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

            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

            del G_rand
            print(f"  [{i+1}/{args.count}] sample written (degree_ok={matches})", file=sys.stderr)

    print(f"Saved {args.count} sample(s) to: {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
