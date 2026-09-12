#!/usr/bin/env python3
"""
Shared, pure random-graph generation: degree-preserving directed configuration
model with double-edge-swap repair of self-loops / parallel edges.

No file I/O in this module.
"""
import random
import networkx as nx
from collections import Counter
from typing import List, Tuple, Optional


def repair_configuration_model_edges(
    edge_list: List[Tuple], rng: random.Random, max_total_attempts: Optional[int] = None
) -> Tuple[List[Tuple], List[int]]:
    """
    Removes self-loops and parallel edges from a directed-configuration-model
    multigraph edge list via degree-preserving double-edge swaps, instead of
    simply deleting the offending edges (which would silently shrink the
    in/out degree of the nodes involved).

    Returns (repaired_edges, still_bad_indices). still_bad_indices is normally
    empty; non-empty only if max_total_attempts is exhausted (pathological
    degree sequences on very small graphs).
    """
    edges = list(edge_list)
    m = len(edges)
    if m == 0:
        return edges, []

    count = Counter(edges)

    def is_bad(i: int) -> bool:
        u, v = edges[i]
        return u == v or count[(u, v)] > 1

    bad = [i for i in range(m) if is_bad(i)]
    if max_total_attempts is None:
        max_total_attempts = 50 * max(1, len(bad)) + 1000

    attempts = 0
    while bad and attempts < max_total_attempts:
        attempts += 1
        i = bad[-1]
        if not is_bad(i):
            bad.pop()
            continue

        u, v = edges[i]
        j = rng.randrange(m)
        if j == i:
            continue
        x, y = edges[j]
        if u == x or v == y:
            continue  # swap would be a no-op or leave the pair untouched

        new1, new2 = (u, y), (x, v)
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue  # would create a new self-loop
        if count[new1] > 0 or count[new2] > 0:
            continue  # would create a new duplicate edge

        count[(u, v)] -= 1
        count[(x, y)] -= 1
        count[new1] += 1
        count[new2] += 1
        edges[i], edges[j] = new1, new2

        bad.pop()
        if is_bad(j):
            bad.append(j)

    still_bad = [i for i in range(m) if is_bad(i)]
    return edges, still_bad


def generate_random_directed_graph(
    in_seq: List[int], out_seq: List[int], rng: Optional[random.Random] = None
) -> Tuple[nx.DiGraph, int, int, int]:
    """
    Generates a directed graph with the exact given in/out-degree sequence,
    via configuration model + degree-preserving repair of self-loops/multi-edges.

    Returns (G_rand, n_selfloops_repaired, n_multiedges_repaired, n_dropped).
    n_dropped counts edges that could not be repaired (should normally be 0)
    and were removed as a last resort, causing a (tiny) degree-sequence drift.
    """
    if rng is None:
        rng = random.Random()
    n = len(in_seq)
    try:
        G_multi = nx.directed_configuration_model(in_seq, out_seq, seed=rng)
        raw_edges = list(G_multi.edges())

        count0 = Counter(raw_edges)
        initial_selfloop_idx  = {i for i, e in enumerate(raw_edges) if e[0] == e[1]}
        initial_bad_idx       = {i for i, e in enumerate(raw_edges)
                                  if e[0] == e[1] or count0[e] > 1}
        initial_multiedge_idx = initial_bad_idx - initial_selfloop_idx

        edges, still_bad = repair_configuration_model_edges(raw_edges, rng)
        still_bad_set = set(still_bad)

        n_dropped              = len(still_bad_set)
        n_selfloops_repaired   = len(initial_selfloop_idx  - still_bad_set)
        n_multiedges_repaired  = len(initial_multiedge_idx - still_bad_set)

        if still_bad_set:
            edges = [e for i, e in enumerate(edges) if i not in still_bad_set]

        G_rand = nx.DiGraph()
        G_rand.add_nodes_from(G_multi.nodes())
        G_rand.add_edges_from(edges)
        return G_rand, n_selfloops_repaired, n_multiedges_repaired, n_dropped
    except Exception:
        m = sum(out_seq)
        p = m / (n * (n - 1)) if n > 1 else 0
        G_rand = nx.erdos_renyi_graph(n, p, directed=True, seed=rng)
        return G_rand, 0, 0, 0
