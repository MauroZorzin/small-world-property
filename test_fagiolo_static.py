#!/usr/bin/env python3
"""
Static correctness tests for fagiolo_clustering_fast() in pipeline.py.

Every expected value in this file is derived by hand from the Fagiolo (2007)
formulas and can be verified with pencil and paper. 

Fagiolo (2007) formulas used
─────────────────────────────
Let A be the binary adjacency matrix (A[i,j]=1 ⟺ edge i->j).
  d_in_i  = Σ_j A[j,i]          (column sum = in-degree)
  d_out_i = Σ_j A[i,j]          (row sum    = out-degree)
  d_tot_i = d_in_i + d_out_i
  d_bil_i = #{j : A[i,j]=1 and A[j,i]=1}   (bilateral / reciprocal degree)

  c_cycle_i      = (A³)[i,i]           / (d_in_i  * d_out_i - d_bil_i)
  c_middleman_i  = (A·Aᵀ·A)[i,i]       / (d_in_i  * d_out_i - d_bil_i)
  c_in_i         = (Aᵀ·A²)[i,i]        / (d_in_i  * (d_in_i  - 1))
  c_out_i        = (A²·Aᵀ)[i,i]        / (d_out_i * (d_out_i - 1))
  c_overall_i    = (A+Aᵀ)³[i,i]        / (2*(d_tot_i*(d_tot_i-1) - 2*d_bil_i))

Each denominator = 0 -> coefficient = 0 (by convention).
"""

import sys
import numpy as np
import networkx as nx
from scipy import sparse
from pipeline import FagioloClusteringAnalyzer

ATOL = 1e-5


# ──────────────────────────────────────────────────────────────────────────────
# Injector helper (no .dot file needed)
# ──────────────────────────────────────────────────────────────────────────────

def make_analyzer(G: nx.DiGraph) -> FagioloClusteringAnalyzer:
    obj = FagioloClusteringAnalyzer.__new__(FagioloClusteringAnalyzer)
    obj.G = G
    obj.n = G.number_of_nodes()
    obj.m = G.number_of_edges()
    obj.original_nodes = list(G.nodes())
    obj.node_to_idx = {n: i for i, n in enumerate(obj.original_nodes)}
    obj.idx_to_node = {i: n for n, i in obj.node_to_idx.items()}
    if obj.n > 0:
        obj.A = nx.to_scipy_sparse_array(G, nodelist=obj.original_nodes,
                                         format='csr', dtype=np.float32)
    else:
        obj.A = sparse.csr_array((0, 0), dtype=np.float32)
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Assertion helpers
# ──────────────────────────────────────────────────────────────────────────────

passed = failed = 0

def check(label, got, expected):
    global passed, failed
    ok = abs(got - expected) < ATOL
    status = "ture" if ok else "false"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {status}  {label}: got {got:.6f}, expected {expected:.6f}")

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1   Pure directed 3-cycle:  0->1->2->0
# ══════════════════════════════════════════════════════════════════════════════
#
# Adjacency matrix (row=from, col=to):
#
#       0  1  2
#   0 [ 0  1  0 ]
#   1 [ 0  0  1 ]
#   2 [ 1  0  0 ]
#
# For every node (by symmetry all nodes are identical):
#   d_in = 1, d_out = 1, d_bil = 0
#
# A² = A@A:
#   (A²)[i,j] counts 2-step paths i->?->j
#   0->1->2: A²[0,2]=1   1->2->0: A²[1,0]=1   2->0->1: A²[2,1]=1
#   All other entries = 0.
#
# A³ = A²@A:
#   Only 3-cycle closes back to start: A³[0,0]=A³[1,1]=A³[2,2]=1
#
# c_cycle = A³[i,i] / (d_in*d_out - d_bil) = 1 / (1*1-0) = 1.0  Yes
#
# Aᵀ@A (for c_middleman numerator = A@Aᵀ@A):
#   Middleman at i means: j->i->k AND j->k directly.
#   No node j has both j->i and j->k for any k reachable from i.
#   -> c_middleman = 0 for all nodes.
#
# c_in: need two distinct predecessors j,k of i with j->k.
#   Each node has exactly one predecessor -> denominator = 1*(1-1)=0 -> c_in=0.
#
# c_out: need two distinct successors j,k of i with j->k.
#   Each node has exactly one successor -> denominator = 1*(1-1)=0 -> c_out=0.
#
# c_overall = (A+Aᵀ)³[i,i] / (2*(d_tot*(d_tot-1) - 2*d_bil))
#   d_tot = 2, d_bil = 0
#   denominator = 2*(2*1 - 0) = 4
#   A+Aᵀ is the symmetric version (undirected triangle).
#   (A+Aᵀ) has every off-diagonal = 1 in the 3-node complete graph.
#   (A+Aᵀ)²[i,i] = sum_j (A+Aᵀ)[i,j]² = 1²+1² = 2
#   (A+Aᵀ)³[i,i] = sum_j (A+Aᵀ)[i,j]*(A+Aᵀ)²[j,i]
#                 = 1*2 + 1*2 = 4   (two paths back through the two neighbors)
#   A+Aᵀ for the 3-cycle is:
#       [ 0  1  1 ]
#       [ 1  0  1 ]
#       [ 1  1  0 ]
#   (A+Aᵀ)²[i,i] = Σ_j (A+Aᵀ)[i,j]^2 = 1+1 = 2  (two neighbors, weight 1 each)
#   (A+Aᵀ)³[i,i] = Σ_j (A+Aᵀ)[i,j] * (A+Aᵀ)²[j,i]
#                 = 1*((A+Aᵀ)²[1,0]) + 1*((A+Aᵀ)²[2,0])
#   (A+Aᵀ)²[j,i] for j≠i: = Σ_k (A+Aᵀ)[j,k]*(A+Aᵀ)[k,i]
#                          = (A+Aᵀ)[j,i_neighbor1]*1 + (A+Aᵀ)[j,i_neighbor2]*1
#   For a 3-node complete undirected graph, (A+Aᵀ)²:
#       diagonal = 2 (two neighbors each with weight 1)
#       off-diag = 1 (one shared neighbor)
#   So (A+Aᵀ)³[i,i] = Σ_j≠i 1 * 1 = 2  (two off-diagonal entries, each = 1)
#   -> c_overall = 2 / 4 = 0.5

section("TEST 1   Pure directed 3-cycle  0->1->2->0")
G = nx.DiGraph([(0,1),(1,2),(2,0)])
az = make_analyzer(G)
r, avg = az.fagiolo_clustering_fast()

# All nodes are symmetric   check all three
for i, name in enumerate(["node 0","node 1","node 2"]):
    check(f"c_cycle     {name}", r['cycle'][i],     1.0)
    check(f"c_middleman {name}", r['middleman'][i],  0.0)
    check(f"c_in        {name}", r['in'][i],         0.0)
    check(f"c_out       {name}", r['out'][i],        0.0)
    check(f"c_overall   {name}", r['overall'][i],    0.5)

check("avg c_cycle",     avg['cycle'],     1.0)
check("avg c_middleman", avg['middleman'],  0.0)
check("avg c_overall",   avg['overall'],   0.5)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2   Directed path: 0->1->2->3  (no triangles anywhere)
# ══════════════════════════════════════════════════════════════════════════════
#
# No node participates in any triangle of any kind.
# All numerators = 0, so all clustering coefficients = 0.
#
# Nodes with degree < 2 also have denominator = 0
# (e.g. node 0: d_in=0, node 3: d_out=0)   convention forces 0.

section("TEST 2   Directed path  0->1->2->3  (no triangles)")
G = nx.DiGraph([(0,1),(1,2),(2,3)])
az = make_analyzer(G)
r, avg = az.fagiolo_clustering_fast()

for metric in ['cycle','middleman','in','out','overall']:
    for i, name in enumerate(["node 0","node 1","node 2","node 3"]):
        check(f"c_{metric} {name}", r[metric][i], 0.0)
    check(f"avg c_{metric}", avg[metric], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3   Fully bidirectional triangle: 0↔1↔2↔0
# ══════════════════════════════════════════════════════════════════════════════
#
# Edges: 0->1, 1->0, 1->2, 2->1, 0->2, 2->0
#
# For every node (all symmetric):
#   d_in = 2, d_out = 2, d_tot = 4, d_bil = 2
#
# A:
#       0  1  2
#   0 [ 0  1  1 ]
#   1 [ 1  0  1 ]
#   2 [ 1  1  0 ]
#
# A³[i,i]: count closed 3-step directed walks starting and ending at i.
#   From node 0: 0->1->0->... can't return; 0->1->2->0 ture; 0->2->1->0 ture; 0->2->0->... no.
#   = 2 walks for each node.
#
# c_cycle = 2 / (2*2 - 2) = 2/2 = 1.0
#
# c_middleman: (A·Aᵀ·A)[i,i].
#   Counts triples (j,k) with j->i, i->k, j->k.
#   Node 0: neighbors in = {1,2}, neighbors out = {1,2}
#   Pairs (j,k): (1,1) invalid same; (1,2) check 1->2 ture; (2,1) check 2->1 ture; (2,2) invalid.
#   -> 2 valid triples.
#   c_middleman = 2 / (2*2 - 2) = 1.0
#
# c_in: (Aᵀ·A²)[i,i] / (d_in*(d_in-1))
#   Counts triples (j,k) with j->i, k->i, j->k.
#   Node 0: in-neighbors = {1,2}. Check 1->2 ture; 2->1 ture -> 2 triples.
#   denominator = 2*(2-1) = 2
#   c_in = 2/2 = 1.0
#
# c_out: (A²·Aᵀ)[i,i] / (d_out*(d_out-1))
#   Counts triples (j,k) with i->j, i->k, j->k.
#   Node 0: out-neighbors = {1,2}. Check 1->2 ture; 2->1 ture -> 2 triples.
#   denominator = 2*(2-1) = 2
#   c_out = 2/2 = 1.0
#
# c_overall: (A+Aᵀ)³[i,i] / (2*(d_tot*(d_tot-1) - 2*d_bil))
#   A+Aᵀ = A (already symmetric, but weights double on bidirectional edges):
#       [ 0  2  2 ]
#       [ 2  0  2 ]
#       [ 2  2  0 ]
#   (A+Aᵀ)²[i,i] = 2²+2² = 8
#   (A+Aᵀ)³[i,i] = Σ_j (A+Aᵀ)[i,j] * (A+Aᵀ)²[j,i]
#   (A+Aᵀ)²[j,i] for j≠i:
#     = Σ_k (A+Aᵀ)[j,k]*(A+Aᵀ)[k,i]
#     = 2*0 + 2*2 = 4   (k=j contributes 0 diagonal; k=third_node contributes 2*2)
#   (A+Aᵀ)³[i,i] = 2*4 + 2*4 = 16
#   denominator = 2*(4*3 - 2*2) = 2*(12-4) = 16
#   c_overall = 16/16 = 1.0

section("TEST 3   Fully bidirectional triangle  0↔1↔2↔0")
G = nx.DiGraph([(0,1),(1,0),(1,2),(2,1),(0,2),(2,0)])
az = make_analyzer(G)
r, avg = az.fagiolo_clustering_fast()

for i, name in enumerate(["node 0","node 1","node 2"]):
    check(f"c_cycle     {name}", r['cycle'][i],     1.0)
    check(f"c_middleman {name}", r['middleman'][i],  1.0)
    check(f"c_in        {name}", r['in'][i],         1.0)
    check(f"c_out       {name}", r['out'][i],        1.0)
    check(f"c_overall   {name}", r['overall'][i],    1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4   Pure out-star:  0->1, 0->2, 0->3
# ══════════════════════════════════════════════════════════════════════════════
#
# No triangles of any kind exist.
# Node 0: d_out=3, d_in=0, d_bil=0
#   c_cycle:     denom = 0*3-0 = 0  -> 0
#   c_middleman: denom = 0*3-0 = 0  -> 0
#   c_in:        denom = 0*(0-1)=0  -> 0
#   c_out:       denom = 3*(3-1)=6, but A²·Aᵀ[0,0]=0 (no out-neighbors connect) -> 0
#   c_overall:   d_tot=3, denom=2*(3*2-0)=12, (A+Aᵀ)³[0,0]=0 (leaves have no edges between them) -> 0
#
# Leaf nodes (1,2,3): d_in=1, d_out=0
#   All denoms = 0 -> all coefficients = 0

section("TEST 4   Pure out-star  0->{1,2,3}  (no triangles)")
G = nx.DiGraph([(0,1),(0,2),(0,3)])
az = make_analyzer(G)
r, avg = az.fagiolo_clustering_fast()

for metric in ['cycle','middleman','in','out','overall']:
    for i in range(4):
        check(f"c_{metric} node {i}", r[metric][i], 0.0)
    check(f"avg c_{metric}", avg[metric], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5   Single bidirectional edge:  0↔1
# ══════════════════════════════════════════════════════════════════════════════
#
# Both nodes: d_in=1, d_out=1, d_bil=1
#   c_cycle:     denom = 1*1-1 = 0 -> 0
#   c_middleman: denom = 1*1-1 = 0 -> 0
#   c_in:        denom = 1*0   = 0 -> 0
#   c_out:       denom = 1*0   = 0 -> 0
#   c_overall:   d_tot=2, denom=2*(2*1-2*1)=2*(2-2)=0 -> 0

section("TEST 5   Single bidirectional edge  0↔1  (all denoms = 0)")
G = nx.DiGraph([(0,1),(1,0)])
az = make_analyzer(G)
r, avg = az.fagiolo_clustering_fast()

for metric in ['cycle','middleman','in','out','overall']:
    for i in range(2):
        check(f"c_{metric} node {i}", r[metric][i], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6   Mixed graph: one cycle triangle + one fan triangle
#
# Nodes: 0,1,2,3
# Edges: 0->1, 1->2, 2->0   (pure directed cycle through 0,1,2)
#        0->3, 1->3          (0 and 1 both point to 3   out-fan at 0 and 1)
# ══════════════════════════════════════════════════════════════════════════════
#
# A (4×4):
#       0  1  2  3
#   0 [ 0  1  0  1 ]
#   1 [ 0  0  1  1 ]
#   2 [ 1  0  0  0 ]
#   3 [ 0  0  0  0 ]
#
# d_in:  node0=1, node1=1, node2=1, node3=2
# d_out: node0=2, node1=2, node2=1, node3=0
# d_bil: no bidirectional edges -> all 0
#
# ── c_cycle ──────────────────────────────────────────────────────────────────
#   Need closed 3-cycles i->j->k->i.
#   Only existing cycle: 0->1->2->0.
#   Node 0: 1 cycle (0->1->2->0). denom = d_in*d_out - d_bil = 1*2-0 = 2. c_cycle = 1/2 = 0.5
#   Node 1: 1 cycle (1->2->0->1). denom = 1*2-0 = 2.                       c_cycle = 1/2 = 0.5
#   Node 2: 1 cycle (2->0->1->2). denom = 1*1-0 = 1.                       c_cycle = 1/1 = 1.0
#   Node 3: no cycle, denom = 2*0-0 = 0.                                 c_cycle = 0
#
# ── c_middleman ──────────────────────────────────────────────────────────────
#   Counts (j,k): j->i, i->k, j->k.
#   Node 0 (in-neighbor={2}, out-neighbors={1,3}):
#     j=2: check 2->1? No. 2->3? No. -> 0
#   Node 1 (in-neighbor={0}, out-neighbors={2,3}):
#     j=0: check 0->2? No. 0->3? Yes ture. -> 1 triple.
#     c_mid = 1/(1*2-0) = 0.5
#   Node 2 (in-neighbor={1}, out-neighbor={0}):
#     j=1: check 1->0? No. -> 0
#   Node 3 (in-neighbors={0,1}, d_out=0): denom=2*0-0=0. c_mid=0
#
# ── c_in ─────────────────────────────────────────────────────────────────────
#   Counts (j,k): j->i, k->i, j->k  (two distinct predecessors where one -> other)
#   Nodes with d_in < 2: denominator=0 -> c_in=0.  (nodes 0,1,2 all have d_in=1)
#   Node 3 (in-neighbors={0,1}): check 0->1 ture and 1->0? No.
#     -> 1 ordered pair (j=0,k=1) where 0->1.
#     denom = 2*(2-1) = 2.
#     c_in = 1/2 = 0.5
#
# ── c_out ────────────────────────────────────────────────────────────────────
#   Counts (j,k): i->j, i->k, j->k  (two distinct successors where one -> other)
#   Nodes with d_out < 2: denominator=0 -> c_out=0.  (nodes 2,3 have d_out<2)
#   Node 0 (out-neighbors={1,3}): check 1->3 ture; check 3->1? No. -> 1 pair.
#     denom = 2*(2-1) = 2.
#     c_out = 1/2 = 0.5
#   Node 1 (out-neighbors={2,3}): check 2->3? No; check 3->2? No. -> 0 pairs.
#     c_out = 0/2 = 0.0
#
# ── c_overall ────────────────────────────────────────────────────────────────
#   (A+Aᵀ)³[i,i] / (2*(d_tot*(d_tot-1) - 2*d_bil))
#
#   A+Aᵀ (undirected adjacency with doubled bidirectional, here all weight 1
#          because no bidirectional edges):
#       0  1  2  3
#   0 [ 0  1  1  1 ]
#   1 [ 1  0  1  1 ]
#   2 [ 1  1  0  0 ]
#   3 [ 1  1  0  0 ]
#
#   (A+Aᵀ)² diagonal counts 2-step return walks = number of neighbors:
#     node0: neighbors {1,2,3} -> 3 neighbors -> (A+Aᵀ)²[0,0] = 3
#     node1: neighbors {0,2,3} -> 3            -> (A+Aᵀ)²[1,1] = 3
#     node2: neighbors {0,1}   -> 2            -> (A+Aᵀ)²[2,2] = 2
#     node3: neighbors {0,1}   -> 2            -> (A+Aᵀ)²[3,3] = 2
#
#   (A+Aᵀ)³[i,i] = Σ_j (A+Aᵀ)[i,j] * (A+Aᵀ)²[j,i]
#   We need (A+Aᵀ)²[j,i] = number of 2-step paths from j to i in the sym graph.
#
#   Full (A+Aᵀ)² matrix (compute row by row):
#     B = A+Aᵀ  (written above)
#     B²[i,j] = Σ_k B[i,k]*B[k,j]
#
#     B²[0,0]: Σ_k B[0,k]*B[k,0] = B[0,1]*B[1,0]+B[0,2]*B[2,0]+B[0,3]*B[3,0]
#             = 1*1+1*1+1*1 = 3
#     B²[0,1]: = B[0,2]*B[2,1]+B[0,3]*B[3,1]+B[0,0]*B[0,1](=0)
#             = 1*1+1*1 = 2
#     B²[0,2]: = B[0,1]*B[1,2]+B[0,0]*B[0,2](=0)+B[0,3]*B[3,2]
#             = 1*1+1*0 = 1
#     B²[0,3]: = B[0,1]*B[1,3]+B[0,0]*B[0,3](=0)+B[0,2]*B[2,3]
#             = 1*1+1*0 = 1
#
#     B²[1,0]: by symmetry = B²[0,1] = 2
#     B²[1,1]: = B[1,0]*B[0,1]+B[1,2]*B[2,1]+B[1,3]*B[3,1] = 1+1+1 = 3
#     B²[1,2]: = B[1,0]*B[0,2]+B[1,3]*B[3,2] = 1*1+1*0 = 1
#     B²[1,3]: = B[1,0]*B[0,3]+B[1,2]*B[2,3] = 1*1+1*0 = 1
#
#     B²[2,0]: by symmetry = B²[0,2] = 1
#     B²[2,1]: by symmetry = B²[1,2] = 1
#     B²[2,2]: = B[2,0]*B[0,2]+B[2,1]*B[1,2] = 1*1+1*1 = 2
#     B²[2,3]: = B[2,0]*B[0,3]+B[2,1]*B[1,3] = 1*1+1*1 = 2
#
#     B²[3,0]: by symmetry = 1
#     B²[3,1]: by symmetry = 1
#     B²[3,2]: by symmetry = B²[2,3] = 2
#     B²[3,3]: = B[3,0]*B[0,3]+B[3,1]*B[1,3] = 1+1 = 2
#
#   (A+Aᵀ)³[i,i] = Σ_j B[i,j] * B²[j,i]
#
#   Node 0: Σ over neighbors {1,2,3}:
#     B[0,1]*B²[1,0] + B[0,2]*B²[2,0] + B[0,3]*B²[3,0]
#     = 1*2 + 1*1 + 1*1 = 4
#   denom0 = 2*(3*2 - 2*0) = 12  ->  c_overall_0 = 4/12 = 1/3
#
#   Node 1: Σ over neighbors {0,2,3}:
#     B[1,0]*B²[0,1] + B[1,2]*B²[2,1] + B[1,3]*B²[3,1]
#     = 1*2 + 1*1 + 1*1 = 4
#   denom1 = 2*(3*2 - 0) = 12  ->  c_overall_1 = 4/12 = 1/3
#
#   Node 2: Σ over neighbors {0,1}:
#     B[2,0]*B²[0,2] + B[2,1]*B²[1,2]
#     = 1*1 + 1*1 = 2
#   denom2 = 2*(2*1 - 0) = 4  ->  c_overall_2 = 2/4 = 0.5
#
#   Node 3: Σ over neighbors {0,1}:
#     B[3,0]*B²[0,3] + B[3,1]*B²[1,3]
#     = 1*1 + 1*1 = 2
#   denom3 = 2*(2*1 - 0) = 4  ->  c_overall_3 = 2/4 = 0.5

section("TEST 6   Mixed graph: cycle 0->1->2->0 plus fans 0->3, 1->3")
G = nx.DiGraph([(0,1),(1,2),(2,0),(0,3),(1,3)])
az = make_analyzer(G)
r, avg = az.fagiolo_clustering_fast()

check("c_cycle     node 0", r['cycle'][0],      0.5)
check("c_cycle     node 1", r['cycle'][1],      0.5)
check("c_cycle     node 2", r['cycle'][2],      1.0)
check("c_cycle     node 3", r['cycle'][3],      0.0)

check("c_middleman node 0", r['middleman'][0],  0.0)
check("c_middleman node 1", r['middleman'][1],  0.5)
check("c_middleman node 2", r['middleman'][2],  0.0)
check("c_middleman node 3", r['middleman'][3],  0.0)

check("c_in        node 0", r['in'][0],         0.0)
check("c_in        node 1", r['in'][1],         0.0)
check("c_in        node 2", r['in'][2],         0.0)
check("c_in        node 3", r['in'][3],         0.5)

check("c_out       node 0", r['out'][0],        0.5)
check("c_out       node 1", r['out'][1],        0.0)
check("c_out       node 2", r['out'][2],        0.0)
check("c_out       node 3", r['out'][3],        0.0)

check("c_overall   node 0", r['overall'][0],    1/3)
check("c_overall   node 1", r['overall'][1],    1/3)
check("c_overall   node 2", r['overall'][2],    0.5)
check("c_overall   node 3", r['overall'][3],    0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  Results: {passed}/{passed+failed} tests passed")
if failed == 0:
    print("  All tests passed ture")
else:
    print(f"  {failed} test(s) FAILED ")
print(f"{'═'*60}\n")

sys.exit(0 if failed == 0 else 1)
