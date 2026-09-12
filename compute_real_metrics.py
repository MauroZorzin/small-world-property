#!/usr/bin/env python3
"""
Loads a Depends .dot file ONCE and computes exact metrics for the real graph
Writes two artifacts

  - <out-dir>/real_metrics.csv       one row with basic/path/clustering/scc stats
  - <out-dir>/degree_sequence.json   in_seq and out_seq
                                      the only thing generate_random_sample.py needs
                                      so it never has to re-parse the slow
                                      memory-heavy .dot file itself

Usage
    python compute_real_metrics.py <project>.dot --out-dir results/<project>/
"""
import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import metrics_logic as ml

# Bump this whenever a code change alters what columns real_metrics.csv gets
# such as a new metric a renamed column or a fixed bug in a value
# An existing file stamped with an older version is treated as stale
# and recomputed automatically so no one has to remember to manually delete
# real_metrics.csv after a code change
SCHEMA_VERSION = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dot_path', type=str, help='Input .dot file')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    dot_path = Path(args.dot_path)
    out_dir = Path(args.out_dir)
    real_metrics_path = out_dir / 'real_metrics.csv'

    if real_metrics_path.exists():
        try:
            existing_version = pd.read_csv(real_metrics_path)['schema_version'].iloc[0]
        except (KeyError, IndexError, pd.errors.EmptyDataError):
            existing_version = None
        if existing_version == SCHEMA_VERSION:
            print(f"{real_metrics_path} already up to date at schema v{SCHEMA_VERSION} "
                  f"skipping the dot file not touched", file=sys.stderr)
            return
        print(f"{real_metrics_path} is stale schema v{existing_version} vs v{SCHEMA_VERSION} "
              f"recomputing", file=sys.stderr)

    if not dot_path.exists():
        print(f"Error: '{dot_path}' not found", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph from {dot_path}", file=sys.stderr)
    G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(str(dot_path)))
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"  Loaded {n} nodes and {m} edges", file=sys.stderr)

    results = {}

    in_seq  = [d for _, d in G.in_degree()]
    out_seq = [d for _, d in G.out_degree()]

    print("Computing basic metrics", file=sys.stderr)
    results.update({f'basic_{k}': v for k, v in ml.basic_metrics(G).items()})
    results['in_degree_median']  = float(np.median(in_seq)) if in_seq else 0.0
    results['out_degree_median'] = float(np.median(out_seq)) if out_seq else 0.0

    print("Computing path lengths for lscc allscc and all_reachable", file=sys.stderr)
    for k, v in ml.all_path_lengths(G).items():
        results[f'path_length_{k}'] = v

    scc_stats = ml.scc_coverage_stats(G)
    results.update({f'scc_{k}': v for k, v in scc_stats.items()})

    print("Computing Fagiolo clustering for full lscc and allscc", file=sys.stderr)
    clustering = ml.fagiolo_scoped(G)
    for k in ml.C_KEYS:
        results[f'clustering_full_{k}']   = clustering['full'][k]
        results[f'clustering_lscc_{k}']   = clustering['lscc'][k]
        results[f'clustering_allscc_{k}'] = clustering['allscc'][k]

    results['schema_version'] = SCHEMA_VERSION
    pd.DataFrame([results]).to_csv(real_metrics_path, index=False)
    print(f"Saved: {real_metrics_path}", file=sys.stderr)

    # Degree sequence is the only thing generate_random_sample.py needs
    degree_seq_path = out_dir / 'degree_sequence.json'
    with open(degree_seq_path, 'w', encoding='utf-8') as f:
        json.dump({'num_nodes': n, 'num_edges': m, 'in_seq': in_seq, 'out_seq': out_seq}, f)
    print(f"Saved: {degree_seq_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
