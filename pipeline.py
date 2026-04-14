#!/usr/bin/env python3
"""
Directed Graph Analysis CLI Tool with Fagiolo Clustering
Usage: python pipeline_v2.py input.dot output.csv [--per-node output_nodes.csv] [--random-iterations N]

Changes from v1
───────────────
* Three distinct path-length variants:
    L_lscc       — average path length on the Largest Strongly Connected Component
    L_allscc     — pairs-weighted average across ALL non-trivial SCCs
    L_undirected — average path length on the undirected version (largest WCC)
* All five Fagiolo clustering variants used as C in sigma:
    overall / cycle / middleman / in / out
* 15 sigma values produced (3 L x 5 C), all stored in the output CSV.
* SCC coverage statistics exported so the paper can report what fraction of
  the graph contributes to each L variant.
"""
import re
import networkx as nx
import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

C_KEYS = ['overall', 'cycle', 'middleman', 'in', 'out']
L_KEYS = ['lscc', 'allscc', 'undirected']


class FagioloClusteringAnalyzer:
    """
    Analyzer for directed graphs implementing Fagiolo (2007) clustering
    coefficients and small-world sigma across multiple L/C variants.
    """

    def __init__(self, graph_path: str):
        """Load graph from .dot file."""
        print(f"Loading graph from {graph_path}...", file=sys.stderr)
        self.G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(graph_path))
        self.n = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()

        self.original_nodes = list(self.G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.original_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        self.A = nx.to_scipy_sparse_array(
            self.G, nodelist=self.original_nodes, format='csr', dtype=np.float32
        )
        print(f"  Loaded: {self.n} nodes, {self.m} edges", file=sys.stderr)

    # ── Basic graph metrics ───────────────────────────────────────────────────

    def basic_metrics(self) -> Dict:
        in_degrees  = np.array([d for _, d in self.G.in_degree()])
        out_degrees = np.array([d for _, d in self.G.out_degree()])
        return {
            'num_nodes':   self.n,
            'num_edges':   self.m,
            'density':     self.m / (self.n * (self.n - 1)) if self.n > 1 else 0,
            'avg_in_degree':  float(np.mean(in_degrees)),
            'avg_out_degree': float(np.mean(out_degrees)),
            'std_in_degree':  float(np.std(in_degrees)),
            'std_out_degree': float(np.std(out_degrees)),
            'max_in_degree':  int(np.max(in_degrees)),
            'max_out_degree': int(np.max(out_degrees)),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.G),
            'num_weakly_connected_components':   nx.number_weakly_connected_components(self.G),
        }

    # ── Three path-length variants ────────────────────────────────────────────

    def path_length_lscc(self) -> float:
        """L on the Largest Strongly Connected Component only."""
        try:
            if nx.is_strongly_connected(self.G):
                return nx.average_shortest_path_length(self.G)
            sccs = list(nx.strongly_connected_components(self.G))
            largest = max(sccs, key=len)
            if len(largest) > 1:
                return nx.average_shortest_path_length(self.G.subgraph(largest))
        except Exception:
            pass
        return float('inf')

    def path_length_allscc(self) -> float:
        """L as a pairs-weighted average across ALL non-trivial SCCs.

        L_allscc = sum_k [n_k*(n_k-1) * L_k] / sum_k [n_k*(n_k-1)]

        Singleton SCCs (n=1) contribute zero pairs and are skipped.
        """
        try:
            if nx.is_strongly_connected(self.G):
                return nx.average_shortest_path_length(self.G)
            sccs = list(nx.strongly_connected_components(self.G))
            non_trivial = [s for s in sccs if len(s) > 1]
            if not non_trivial:
                return float('inf')
            weighted_sum = 0.0
            total_weight = 0.0
            for scc in non_trivial:
                pairs = len(scc) * (len(scc) - 1)
                L_scc = nx.average_shortest_path_length(self.G.subgraph(scc))
                weighted_sum += pairs * L_scc
                total_weight += pairs
            return weighted_sum / total_weight if total_weight > 0 else float('inf')
        except Exception:
            pass
        return float('inf')

    def path_length_undirected(self) -> float:
        """L on the undirected version of the graph (largest WCC).

        Converting to undirected adds the reverse direction for every edge,
        making the graph connected in more cases and providing a lower-bound
        estimate of average distance.
        """
        try:
            G_und = self.G.to_undirected()
            if nx.is_connected(G_und):
                return nx.average_shortest_path_length(G_und)
            wccs = list(nx.connected_components(G_und))
            largest_wcc = max(wccs, key=len)
            if len(largest_wcc) > 1:
                return nx.average_shortest_path_length(G_und.subgraph(largest_wcc))
        except Exception:
            pass
        return float('inf')

    def all_path_lengths(self) -> Dict[str, float]:
        """Return all three L variants in one dict."""
        return {
            'lscc':       self.path_length_lscc(),
            'allscc':     self.path_length_allscc(),
            'undirected': self.path_length_undirected(),
        }

    # ── SCC coverage statistics ───────────────────────────────────────────────

    def scc_coverage_stats(self) -> Dict:
        """Fraction of the full graph captured by non-trivial SCCs."""
        sccs = list(nx.strongly_connected_components(self.G))
        non_trivial = [s for s in sccs if len(s) > 1]
        largest = max(sccs, key=len) if sccs else set()

        nodes_in_sccs = sum(len(s) for s in non_trivial)
        edges_in_sccs = sum(self.G.subgraph(s).number_of_edges() for s in non_trivial)

        return {
            'num_sccs_total':       len(sccs),
            'num_sccs_non_trivial': len(non_trivial),
            'lscc_nodes':           len(largest),
            'lscc_edges':           self.G.subgraph(largest).number_of_edges(),
            'lscc_node_pct':        round(len(largest) / self.n * 100, 2) if self.n else 0.0,
            'allscc_nodes':         nodes_in_sccs,
            'allscc_edges':         edges_in_sccs,
            'allscc_node_pct':      round(nodes_in_sccs / self.n * 100, 2) if self.n else 0.0,
            'allscc_edge_pct':      round(edges_in_sccs / self.m * 100, 2) if self.m else 0.0,
        }

    # ── Bilateral edges ───────────────────────────────────────────────────────

    def compute_bilateral_edges_fast(self) -> np.ndarray:
        bilateral_matrix = self.A.multiply(self.A.T)
        return np.array(bilateral_matrix.sum(axis=1)).flatten()

    # ── Fagiolo clustering (all five variants) ────────────────────────────────

    def fagiolo_clustering_fast(self) -> Tuple[Dict, Dict]:
        """
        Vectorised Fagiolo (2007) directed clustering coefficients.
        Returns (per_node_dict, averages_dict).

        Variants
        --------
        overall    (A+At)^3[i,i]  / (2*(d_tot*(d_tot-1) - 2*d_bil))
        cycle      A^3[i,i]        / (d_in*d_out - d_bil)
        middleman  (A*At*A)[i,i]  / (d_in*d_out - d_bil)
        in         (At*A^2)[i,i]  / (d_in*(d_in-1))
        out        (A^2*At)[i,i]  / (d_out*(d_out-1))
        """
        print("Computing Fagiolo clustering coefficients...", file=sys.stderr)

        A  = self.A
        AT = A.T

        d_in  = np.array(A.sum(axis=0)).flatten()
        d_out = np.array(A.sum(axis=1)).flatten()
        d_tot = d_in + d_out

        d_bilateral = self.compute_bilateral_edges_fast()

        A2     = A  @ A
        A3     = A2 @ A
        AAT_A  = A  @ AT @ A
        AT_A2  = AT @ A2
        A2_AT  = A2 @ AT
        A_sym  = A  + AT
        A_sym3 = A_sym @ A_sym @ A_sym

        diag_A3     = np.array(A3.diagonal()).flatten()
        diag_AAT_A  = np.array(AAT_A.diagonal()).flatten()
        diag_AT_A2  = np.array(AT_A2.diagonal()).flatten()
        diag_A2_AT  = np.array(A2_AT.diagonal()).flatten()
        diag_A_sym3 = np.array(A_sym3.diagonal()).flatten()

        denom_cycle     = d_in * d_out - d_bilateral
        denom_middleman = d_in * d_out - d_bilateral
        denom_in        = d_in  * (d_in  - 1)
        denom_out       = d_out * (d_out - 1)
        denom_overall   = 2 * (d_tot * (d_tot - 1) - 2 * d_bilateral)

        def _div(num, den):
            return np.divide(num, den, out=np.zeros_like(num), where=den != 0)

        c_cycle     = _div(diag_A3,     denom_cycle)
        c_middleman = _div(diag_AAT_A,  denom_middleman)
        c_in        = _div(diag_AT_A2,  denom_in)
        c_out       = _div(diag_A2_AT,  denom_out)
        c_overall   = _div(diag_A_sym3, denom_overall)

        per_node = {
            'overall':          c_overall.tolist(),
            'cycle':            c_cycle.tolist(),
            'middleman':        c_middleman.tolist(),
            'in':               c_in.tolist(),
            'out':              c_out.tolist(),
            'in_degree':        d_in.tolist(),
            'out_degree':       d_out.tolist(),
            'total_degree':     d_tot.tolist(),
            'bilateral_degree': d_bilateral.tolist(),
        }

        averages = {}
        for key in C_KEYS:
            vals = [v for v in per_node[key] if np.isfinite(v)]
            averages[key] = float(np.mean(vals)) if vals else 0.0

        return per_node, averages

    # ── Per-node CSV export ───────────────────────────────────────────────────

    def _parse_node_names_from_comments(self, dot_path: str) -> Dict:
        id_to_name = {}
        comment_pattern = re.compile(r"^\s*//\s*(\d+):(.+)$")
        try:
            with open(dot_path, 'r', encoding='utf-8') as f:
                for line in f:
                    m = comment_pattern.match(line)
                    if m:
                        id_to_name[m.group(1)] = m.group(2).strip()
        except Exception as e:
            print(f"Warning: Could not parse comments for node names: {e}", file=sys.stderr)
        return id_to_name

    def export_per_node_metrics(self, clustering_results: Dict,
                                output_path: str, dot_path: str):
        id_to_name = self._parse_node_names_from_comments(dot_path)
        data = []
        for idx, node in enumerate(self.original_nodes):
            nid = str(node)
            data.append({
                'node_id':              nid,
                'node_name':            id_to_name.get(nid, 'Unknown'),
                'in_degree':            int(clustering_results['in_degree'][idx]),
                'out_degree':           int(clustering_results['out_degree'][idx]),
                'total_degree':         int(clustering_results['total_degree'][idx]),
                'bilateral_degree':     int(clustering_results['bilateral_degree'][idx]),
                'clustering_overall':   clustering_results['overall'][idx],
                'clustering_cycle':     clustering_results['cycle'][idx],
                'clustering_middleman': clustering_results['middleman'][idx],
                'clustering_in':        clustering_results['in'][idx],
                'clustering_out':       clustering_results['out'][idx],
            })
        df = pd.DataFrame(data).sort_values('total_degree', ascending=False)
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} node records", file=sys.stderr)
        return df

    # ── Random graph generation ───────────────────────────────────────────────

    def generate_random_directed_graph(self) -> nx.DiGraph:
        in_seq  = [d for _, d in self.G.in_degree()]
        out_seq = [d for _, d in self.G.out_degree()]
        try:
            G_rand = nx.directed_configuration_model(
                in_seq, out_seq, create_using=nx.DiGraph()
            )
            G_rand = nx.DiGraph(G_rand)
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
        except Exception:
            p = self.m / (self.n * (self.n - 1)) if self.n > 1 else 0
            G_rand = nx.erdos_renyi_graph(self.n, p, directed=True)
        return G_rand

    # ── Small-world sigma (15 variants) ──────────────────────────────────────

    def compute_small_worldness(self, num_random: int = 10) -> Dict:
        """
        Compute sigma for every combination of:
            C in {overall, cycle, middleman, in, out}   (5 Fagiolo variants)
            L in {lscc, allscc, undirected}             (3 path-length variants)

        Produces 15 sigma values.  All random graphs are evaluated on all
        variants in a single pass so baselines are internally consistent.

        Output key naming
        -----------------
        C_orig_<c>               original clustering value for variant c
        C_rand_mean_<c>          mean over random graphs
        C_rand_std_<c>           std  over random graphs
        L_orig_<l>               original path length for variant l
        L_rand_mean_<l>          mean over random graphs
        L_rand_std_<l>           std  over random graphs
        sigma_<c>_<l>            small-world coefficient  (C/Crand) / (L/Lrand)
        is_smallworld_<c>_<l>    True when sigma > 1
        """
        print(
            f"Computing small-worldness "
            f"({num_random} random graphs, "
            f"{len(C_KEYS)} C-variants x {len(L_KEYS)} L-variants "
            f"= {len(C_KEYS) * len(L_KEYS)} sigma values)...",
            file=sys.stderr
        )

        # Original values
        _, C_orig = self.fagiolo_clustering_fast()
        L_orig    = self.all_path_lengths()

        # Accumulate random-graph values
        rand_C: Dict[str, list] = {k: [] for k in C_KEYS}
        rand_L: Dict[str, list] = {k: [] for k in L_KEYS}

        for i in range(num_random):
            print(f"  Random graph {i+1}/{num_random}...", end='\r', file=sys.stderr)
            G_rand = self.generate_random_directed_graph()

            r = FagioloClusteringAnalyzer.__new__(FagioloClusteringAnalyzer)
            r.G              = G_rand
            r.n              = G_rand.number_of_nodes()
            r.m              = G_rand.number_of_edges()
            r.original_nodes = list(G_rand.nodes())
            r.node_to_idx    = {node: idx for idx, node in enumerate(r.original_nodes)}
            r.A              = nx.to_scipy_sparse_array(
                G_rand, nodelist=r.original_nodes, format='csr', dtype=np.float32
            )

            _, r_C = r.fagiolo_clustering_fast()
            for k in C_KEYS:
                if np.isfinite(r_C[k]):
                    rand_C[k].append(r_C[k])

            r_L = r.all_path_lengths()
            for k in L_KEYS:
                if np.isfinite(r_L[k]):
                    rand_L[k].append(r_L[k])

        print("", file=sys.stderr)

        # Aggregate
        C_rand_mean = {k: float(np.mean(rand_C[k])) if rand_C[k] else float('nan')
                       for k in C_KEYS}
        C_rand_std  = {k: float(np.std(rand_C[k]))  if len(rand_C[k]) > 1 else 0.0
                       for k in C_KEYS}
        L_rand_mean = {k: float(np.mean(rand_L[k])) if rand_L[k] else float('nan')
                       for k in L_KEYS}
        L_rand_std  = {k: float(np.std(rand_L[k]))  if len(rand_L[k]) > 1 else 0.0
                       for k in L_KEYS}

        def _sigma(c_key: str, l_key: str) -> float:
            C  = C_orig[c_key]
            L  = L_orig[l_key]
            Cr = C_rand_mean[c_key]
            Lr = L_rand_mean[l_key]
            if all(np.isfinite(v) for v in [C, L, Cr, Lr]) and Cr > 0 and Lr > 0:
                return float((C / Cr) / (L / Lr))
            return float('nan')

        # Build output dict
        out: Dict = {}

        for k in C_KEYS:
            out[f'C_orig_{k}']      = C_orig[k]
            out[f'C_rand_mean_{k}'] = C_rand_mean[k]
            out[f'C_rand_std_{k}']  = C_rand_std[k]

        for k in L_KEYS:
            out[f'L_orig_{k}']      = L_orig[k]
            out[f'L_rand_mean_{k}'] = L_rand_mean[k]
            out[f'L_rand_std_{k}']  = L_rand_std[k]

        for c_key in C_KEYS:
            for l_key in L_KEYS:
                s = _sigma(c_key, l_key)
                out[f'sigma_{c_key}_{l_key}']         = s
                out[f'is_smallworld_{c_key}_{l_key}'] = bool(s > 1) if np.isfinite(s) else False

        return out

    # ── Main analysis entry point ─────────────────────────────────────────────

    def analyze_and_export(self, output_path: str, dot_path: str,
                           per_node_path: Optional[str] = None,
                           num_random: int = 10) -> Dict:
        """Run complete analysis and export results to CSV."""
        print("\n" + "=" * 70, file=sys.stderr)
        print("DIRECTED GRAPH ANALYSIS", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)

        results = {}

        # Basic metrics
        print("Computing basic metrics...", file=sys.stderr)
        basic = self.basic_metrics()
        results.update({f'basic_{k}': v for k, v in basic.items()})

        # Three path lengths
        print("Computing path lengths (lscc / allscc / undirected)...", file=sys.stderr)
        path_lengths = self.all_path_lengths()
        for k, v in path_lengths.items():
            results[f'path_length_{k}'] = v

        # SCC coverage
        scc_stats = self.scc_coverage_stats()
        results.update({f'scc_{k}': v for k, v in scc_stats.items()})

        # Degree stats
        print("Computing degree statistics...", file=sys.stderr)
        in_degrees  = [d for _, d in self.G.in_degree()]
        out_degrees = [d for _, d in self.G.out_degree()]
        results['in_degree_median']  = float(np.median(in_degrees))
        results['out_degree_median'] = float(np.median(out_degrees))

        # Fagiolo clustering
        clustering_results, clustering_avg = self.fagiolo_clustering_fast()
        results.update({f'clustering_{k}': v for k, v in clustering_avg.items()})

        # Per-node CSV
        if per_node_path:
            self.export_per_node_metrics(clustering_results, per_node_path, dot_path)

        # Small-worldness (15 sigmas)
        sw = self.compute_small_worldness(num_random)
        results.update({f'smallworld_{k}': v for k, v in sw.items()})

        # Save summary CSV
        pd.DataFrame([results]).to_csv(output_path, index=False)
        print(f"\n  Summary saved to: {output_path}", file=sys.stderr)
        if per_node_path:
            print(f"  Per-node saved to: {per_node_path}", file=sys.stderr)

        # Human-readable summary table
        print("\n" + "=" * 70, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"Nodes: {self.n}  Edges: {self.m}", file=sys.stderr)
        print(
            f"SCC coverage   LSCC: {scc_stats['lscc_node_pct']}% nodes | "
            f"AllSCC: {scc_stats['allscc_node_pct']}% nodes / "
            f"{scc_stats['allscc_edge_pct']}% edges "
            f"({scc_stats['num_sccs_non_trivial']} non-trivial SCCs)",
            file=sys.stderr
        )
        print(
            f"Path lengths   LSCC: {path_lengths['lscc']:.8f} | "
            f"AllSCC: {path_lengths['allscc']:.8f} | "
            f"Undirected: {path_lengths['undirected']:.8f}",
            file=sys.stderr
        )
        print(
            f"Clustering     overall:{clustering_avg['overall']:.8f}  "
            f"cycle:{clustering_avg['cycle']:.8f}  "
            f"mid:{clustering_avg['middleman']:.8f}  "
            f"in:{clustering_avg['in']:.8f}  "
            f"out:{clustering_avg['out']:.8f}",
            file=sys.stderr
        )
        print("Sigma (C \\ L):", file=sys.stderr)
        print(f"  {'':12}" + "".join(f"{lk:>14}" for lk in L_KEYS), file=sys.stderr)
        for ck in C_KEYS:
            row = f"  {ck:12}" + "".join(
                f"{sw.get(f'sigma_{ck}_{lk}', float('nan')):>14.8f}"
                for lk in L_KEYS
            )
            print(row, file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze directed graphs with Fagiolo clustering (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dependency_graph.dot results.csv
  %(prog)s graph.dot summary.csv --per-node nodes.csv
  %(prog)s input.dot output.csv --per-node per_node.csv --random-iterations 20

Output:
  Summary CSV  -- one row, all metrics including 15 sigma values
  Per-node CSV -- one row per class (optional)
        """
    )
    parser.add_argument('input',  type=str, help='Input .dot file')
    parser.add_argument('output', type=str, help='Output summary CSV')
    parser.add_argument('-p', '--per-node', type=str, default=None,
                        help='Output per-node CSV (optional)')
    parser.add_argument('-r', '--random-iterations', type=int, default=10,
                        help='Random graphs for sigma baseline (default: 10)')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_node_path = None
    if args.per_node:
        per_node_path = Path(args.per_node)
        per_node_path.parent.mkdir(parents=True, exist_ok=True)
        per_node_path = str(per_node_path)

    try:
        analyzer = FagioloClusteringAnalyzer(str(input_path))
        analyzer.analyze_and_export(
            str(output_path), str(input_path), per_node_path, args.random_iterations
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()