#!/usr/bin/env python3
"""
Shared, pure graph-metric computations: Fagiolo (2007) directed clustering,
path lengths, SCC coverage, basic degree stats.

No file I/O in this module. Used identically by compute_real_metrics.py (on
the real graph) and generate_random_sample.py (on each random graph), so the
two never duplicate the underlying math.
"""
import networkx as nx
import numpy as np
from typing import Dict, Tuple

C_KEYS   = ['overall', 'cycle', 'middleman', 'in', 'out']
L_KEYS   = ['lscc', 'allscc', 'all_reachable']
C_SCOPES = ['full', 'lscc', 'allscc']


def fagiolo_on_graph(G: nx.DiGraph) -> Tuple[Dict, Dict]:
    """
    Vectorised Fagiolo (2007) directed clustering for a given DiGraph.
    Returns (per_node_dict, averages_dict).
    """
    n = G.number_of_nodes()
    if n == 0:
        empty = {k: float('nan') for k in C_KEYS}
        return {k: [] for k in C_KEYS + ['in_degree', 'out_degree',
                                          'total_degree', 'bilateral_degree']}, empty

    nodes = list(G.nodes())
    A  = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr', dtype=np.float32)
    AT = A.T.tocsr()

    d_in  = np.array(A.sum(axis=0)).flatten()
    d_out = np.array(A.sum(axis=1)).flatten()
    d_tot = d_in + d_out

    bilateral_matrix = A.multiply(AT)
    d_bilateral = np.array(bilateral_matrix.sum(axis=1)).flatten()

    A2  = A @ A
    AAT = A @ AT
    A2T = A2.T.tocsr()

    # Only 2 full sparse products (A2, AAT); the 4 triple-product diagonals are
    # obtained elementwise (diag(M@X)_i = sum_j M[i,j]*X[j,i]) — O(n*d^2) not O(n*d^3).
    diag_A3    = np.array(A2.multiply(AT).sum(axis=1)).flatten()
    diag_AAT_A = np.array(AAT.multiply(AT).sum(axis=1)).flatten()
    diag_AT_A2 = np.array(AT.multiply(A2T).sum(axis=1)).flatten()
    diag_A2_AT = np.array(A2.multiply(A).sum(axis=1)).flatten()
    # (A+AT)^3's diagonal = 2*(sum of the 4 above), Fagiolo eq.13 — A_sym3 never built.
    diag_A_sym3 = 2 * (diag_A3 + diag_AAT_A + diag_AT_A2 + diag_A2_AT)

    denom_cycle     = d_in * d_out - d_bilateral
    denom_middleman = d_in * d_out - d_bilateral
    denom_in        = d_in  * (d_in  - 1)
    denom_out       = d_out * (d_out - 1)
    denom_overall   = 2 * (d_tot * (d_tot - 1) - 2 * d_bilateral)

    def _div(num, den):
        return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)

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
        averages[key] = float(np.mean(vals)) if vals else float('nan')

    return per_node, averages


def allscc_weighted_clustering(G: nx.DiGraph) -> Dict[str, float]:
    """Pairs-weighted average of Fagiolo clustering across ALL non-trivial SCCs."""
    sccs = [s for s in nx.strongly_connected_components(G) if len(s) > 1]
    if not sccs:
        return {k: float('nan') for k in C_KEYS}

    weighted = {k: 0.0 for k in C_KEYS}
    total_weight = 0.0

    for scc in sccs:
        w = len(scc) * (len(scc) - 1)
        _, avg = fagiolo_on_graph(G.subgraph(scc).copy())
        for k in C_KEYS:
            if np.isfinite(avg[k]):
                weighted[k] += w * avg[k]
        total_weight += w

    if total_weight == 0:
        return {k: float('nan') for k in C_KEYS}
    return {k: weighted[k] / total_weight for k in C_KEYS}


def fagiolo_clustering_lscc(G: nx.DiGraph) -> Dict[str, float]:
    """Fagiolo averages computed on the LSCC subgraph only."""
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return {k: float('nan') for k in C_KEYS}
    largest = max(sccs, key=len)
    if len(largest) < 2:
        return {k: float('nan') for k in C_KEYS}
    _, avg = fagiolo_on_graph(G.subgraph(largest).copy())
    return avg


def fagiolo_scoped(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """Returns clustering averages for all three scopes: {'full','lscc','allscc'}."""
    _, full_avg = fagiolo_on_graph(G)
    return {
        'full':   full_avg,
        'lscc':   fagiolo_clustering_lscc(G),
        'allscc': allscc_weighted_clustering(G),
    }


def path_length_lscc(G: nx.DiGraph) -> float:
    try:
        if nx.is_strongly_connected(G):
            return nx.average_shortest_path_length(G)
        sccs = list(nx.strongly_connected_components(G))
        largest = max(sccs, key=len)
        if len(largest) > 1:
            return nx.average_shortest_path_length(G.subgraph(largest))
    except Exception:
        pass
    return float('inf')


def path_length_allscc(G: nx.DiGraph) -> float:
    try:
        if nx.is_strongly_connected(G):
            return nx.average_shortest_path_length(G)
        sccs = list(nx.strongly_connected_components(G))
        non_trivial = [s for s in sccs if len(s) > 1]
        if not non_trivial:
            return float('inf')
        weighted_sum = 0.0
        total_weight = 0.0
        for scc in non_trivial:
            pairs = len(scc) * (len(scc) - 1)
            L_scc = nx.average_shortest_path_length(G.subgraph(scc))
            weighted_sum += pairs * L_scc
            total_weight += pairs
        return weighted_sum / total_weight if total_weight > 0 else float('inf')
    except Exception:
        pass
    return float('inf')


def path_length_all_reachable(G: nx.DiGraph) -> float:
    total = 0
    count = 0
    for node in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, node)
        for target, dist in lengths.items():
            if target != node:
                total += dist
                count += 1
    if count == 0:
        return float('inf')
    return total / count


def all_path_lengths(G: nx.DiGraph) -> Dict[str, float]:
    return {
        'lscc':          path_length_lscc(G),
        'allscc':        path_length_allscc(G),
        'all_reachable': path_length_all_reachable(G),
    }


def scc_coverage_stats(G: nx.DiGraph) -> Dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    sccs = list(nx.strongly_connected_components(G))
    non_trivial = [s for s in sccs if len(s) > 1]
    largest = max(sccs, key=len) if sccs else set()

    nodes_in_sccs = sum(len(s) for s in non_trivial)
    edges_in_sccs = sum(G.subgraph(s).number_of_edges() for s in non_trivial)

    return {
        'num_sccs_total':       len(sccs),
        'num_sccs_non_trivial': len(non_trivial),
        'lscc_nodes':           len(largest),
        'lscc_edges':           G.subgraph(largest).number_of_edges(),
        'lscc_node_pct':        round(len(largest) / n * 100, 2) if n else 0.0,
        'allscc_nodes':         nodes_in_sccs,
        'allscc_edges':         edges_in_sccs,
        'allscc_node_pct':      round(nodes_in_sccs / n * 100, 2) if n else 0.0,
        'allscc_edge_pct':      round(edges_in_sccs / m * 100, 2) if m else 0.0,
    }


def basic_metrics(G: nx.DiGraph) -> Dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    in_degrees  = np.array([d for _, d in G.in_degree()])
    out_degrees = np.array([d for _, d in G.out_degree()])
    return {
        'num_nodes':   n,
        'num_edges':   m,
        'density':     m / (n * (n - 1)) if n > 1 else 0,
        'avg_in_degree':  float(np.mean(in_degrees)) if n else 0.0,
        'avg_out_degree': float(np.mean(out_degrees)) if n else 0.0,
        'std_in_degree':  float(np.std(in_degrees)) if n else 0.0,
        'std_out_degree': float(np.std(out_degrees)) if n else 0.0,
        'max_in_degree':  int(np.max(in_degrees)) if n else 0,
        'max_out_degree': int(np.max(out_degrees)) if n else 0,
        'num_strongly_connected_components': nx.number_strongly_connected_components(G),
        'num_weakly_connected_components':   nx.number_weakly_connected_components(G),
    }
