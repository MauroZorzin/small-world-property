#!/usr/bin/env python3
"""
Directed Graph Analysis CLI Tool with Fagiolo Clustering
Usage: python graph_analyzer.py input.dot output.csv [--random-iterations N]
"""

import networkx as nx
import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class FagioloClusteringAnalyzer:
    """
    Optimized analyzer for directed graphs implementing Fagiolo clustering coefficients.
    Reference: Fagiolo, G. (2007). Clustering in complex directed networks.
    Physical Review E, 76(2), 026107.
    """
    
    def __init__(self, graph_path: str):
        """Load graph from .dot file"""
        print(f"Loading graph from {graph_path}...", file=sys.stderr)
        self.G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(graph_path))
        self.n = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()
        
        # Create node to index mapping for faster matrix operations
        self.node_to_idx = {node: idx for idx, node in enumerate(self.G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Pre-compute adjacency matrix for performance
        self.A = nx.to_scipy_sparse_array(self.G, nodelist=list(self.G.nodes()), 
                                          format='csr', dtype=np.float32)
        
        print(f"  Loaded: {self.n} nodes, {self.m} edges", file=sys.stderr)
        
    def basic_metrics(self) -> Dict:
        """Compute basic graph metrics (optimized)"""
        in_degrees = np.array([d for n, d in self.G.in_degree()])
        out_degrees = np.array([d for n, d in self.G.out_degree()])
        
        metrics = {
            'num_nodes': self.n,
            'num_edges': self.m,
            'density': self.m / (self.n * (self.n - 1)) if self.n > 1 else 0,
            'avg_in_degree': float(np.mean(in_degrees)),
            'avg_out_degree': float(np.mean(out_degrees)),
            'std_in_degree': float(np.std(in_degrees)),
            'std_out_degree': float(np.std(out_degrees)),
            'max_in_degree': int(np.max(in_degrees)),
            'max_out_degree': int(np.max(out_degrees)),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.G),
            'num_weakly_connected_components': nx.number_weakly_connected_components(self.G),
        }
        
        return metrics
    
    def average_path_length(self) -> float:
        """Compute average shortest path length (optimized for directed graphs)"""
        try:
            if nx.is_strongly_connected(self.G):
                return nx.average_shortest_path_length(self.G)
        except:
            pass
        
        # Use largest strongly connected component
        try:
            sccs = list(nx.strongly_connected_components(self.G))
            if sccs:
                largest_scc = max(sccs, key=len)
                if len(largest_scc) > 1:
                    G_scc = self.G.subgraph(largest_scc)
                    return nx.average_shortest_path_length(G_scc)
        except:
            pass
        
        return float('inf')
    
    def compute_bilateral_edges_fast(self) -> np.ndarray:
        """Fast computation of bilateral edges using sparse matrix operations"""
        # Element-wise multiplication of A and A^T to find bidirectional edges
        bilateral_matrix = self.A.multiply(self.A.T)
        # Sum along rows to get bilateral degree for each node
        bilateral_degrees = np.array(bilateral_matrix.sum(axis=1)).flatten()
        return bilateral_degrees
    
    def fagiolo_clustering_fast(self) -> Tuple[Dict, Dict]:
        """
        Optimized computation of all Fagiolo clustering coefficients using vectorized operations.
        This computes all coefficients in a single pass using sparse matrix operations.
        """
        print("Computing Fagiolo clustering coefficients...", file=sys.stderr)
        
        # Pre-compute necessary matrices
        A = self.A
        AT = A.T
        
        # Compute degrees (vectorized)
        d_in = np.array(A.sum(axis=0)).flatten()
        d_out = np.array(A.sum(axis=1)).flatten()
        d_tot = d_in + d_out
        
        # Compute bilateral edges
        d_bilateral = self.compute_bilateral_edges_fast()
        
        # Pre-compute matrix powers and products (sparse operations)
        A2 = A @ A
        A3 = A2 @ A
        AAT_A = A @ AT @ A
        AT_A2 = AT @ A2
        A2_AT = A2 @ AT
        A_sym = A + AT
        A_sym3 = A_sym @ A_sym @ A_sym
        
        # Extract diagonals (these are the numerators)
        diag_A3 = np.array(A3.diagonal()).flatten()
        diag_AAT_A = np.array(AAT_A.diagonal()).flatten()
        diag_AT_A2 = np.array(AT_A2.diagonal()).flatten()
        diag_A2_AT = np.array(A2_AT.diagonal()).flatten()
        diag_A_sym3 = np.array(A_sym3.diagonal()).flatten()
        
        # Compute denominators (vectorized)
        denom_cycle = d_in * d_out - d_bilateral
        denom_middleman = d_in * d_out - d_bilateral
        denom_in = d_in * (d_in - 1)
        denom_out = d_out * (d_out - 1)
        denom_overall = 2 * (d_tot * (d_tot - 1) - 2 * d_bilateral)
        
        # Compute clustering coefficients (avoiding division by zero)
        c_cycle = np.divide(diag_A3, denom_cycle, 
                           out=np.zeros_like(diag_A3), where=denom_cycle!=0)
        c_middleman = np.divide(diag_AAT_A, denom_middleman, 
                               out=np.zeros_like(diag_AAT_A), where=denom_middleman!=0)
        c_in = np.divide(diag_AT_A2, denom_in, 
                        out=np.zeros_like(diag_AT_A2), where=denom_in!=0)
        c_out = np.divide(diag_A2_AT, denom_out, 
                         out=np.zeros_like(diag_A2_AT), where=denom_out!=0)
        c_overall = np.divide(diag_A_sym3, denom_overall, 
                             out=np.zeros_like(diag_A_sym3), where=denom_overall!=0)
        
        # Store results per node
        results = {
            'overall': c_overall.tolist(),
            'cycle': c_cycle.tolist(),
            'middleman': c_middleman.tolist(),
            'in': c_in.tolist(),
            'out': c_out.tolist()
        }
        
        # Compute averages (ignoring inf and nan)
        averages = {}
        for key, values in results.items():
            valid_values = [v for v in values if np.isfinite(v)]
            averages[key] = np.mean(valid_values) if valid_values else 0.0
        
        return results, averages
    
    def generate_random_directed_graph(self) -> nx.DiGraph:
        """Generate random directed graph with same degree sequence"""
        in_seq = [d for n, d in self.G.in_degree()]
        out_seq = [d for n, d in self.G.out_degree()]
        
        try:
            G_random = nx.directed_configuration_model(in_seq, out_seq, 
                                                       create_using=nx.DiGraph())
            G_random = nx.DiGraph(G_random)
            G_random.remove_edges_from(nx.selfloop_edges(G_random))
        except:
            # Fallback to Erdős-Rényi
            p = self.m / (self.n * (self.n - 1)) if self.n > 1 else 0
            G_random = nx.erdos_renyi_graph(self.n, p, directed=True)
        
        return G_random
    
    def compute_small_worldness(self, num_random: int = 10) -> Dict:
        """Compute small-worldness coefficient with random graph comparison"""
        print(f"Computing small-worldness (comparing to {num_random} random graphs)...", 
              file=sys.stderr)
        
        # Compute metrics for original graph
        _, C_orig_dict = self.fagiolo_clustering_fast()
        C_orig = C_orig_dict['overall']
        L_orig = self.average_path_length()
        
        # Compute metrics for random graphs
        C_random_list = []
        L_random_list = []
        
        for i in range(num_random):
            print(f"  Random graph {i+1}/{num_random}...", end='\r', file=sys.stderr)
            G_random = self.generate_random_directed_graph()
            
            # Create temporary analyzer for random graph
            analyzer_random = FagioloClusteringAnalyzer.__new__(FagioloClusteringAnalyzer)
            analyzer_random.G = G_random
            analyzer_random.n = G_random.number_of_nodes()
            analyzer_random.m = G_random.number_of_edges()
            analyzer_random.node_to_idx = {node: idx for idx, node in enumerate(G_random.nodes())}
            analyzer_random.A = nx.to_scipy_sparse_array(G_random, 
                                                         nodelist=list(G_random.nodes()), 
                                                         format='csr', dtype=np.float32)
            
            _, C_random_avg = analyzer_random.fagiolo_clustering_fast()
            C_random_list.append(C_random_avg['overall'])
            L_random_list.append(analyzer_random.average_path_length())
        
        print("", file=sys.stderr)  # New line after progress
        
        C_random = np.mean(C_random_list)
        L_random = np.mean(L_random_list)
        
        # Compute small-worldness
        if C_random > 0 and L_random > 0 and np.isfinite(L_random) and np.isfinite(L_orig):
            sigma = (C_orig / C_random) / (L_orig / L_random)
        else:
            sigma = float('nan')
        
        return {
            'sigma': sigma,
            'C_orig': C_orig,
            'L_orig': L_orig,
            'C_random_mean': C_random,
            'L_random_mean': L_random,
            'C_random_std': np.std(C_random_list),
            'L_random_std': np.std(L_random_list),
            'is_small_world': sigma > 1 if np.isfinite(sigma) else False
        }
    
    def analyze_and_export(self, output_path: str, num_random: int = 10):
        """Run complete analysis and export results to CSV"""
        print("\n" + "="*70, file=sys.stderr)
        print("DIRECTED GRAPH ANALYSIS", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr)
        
        results = {}
        
        # Basic metrics
        print("Computing basic metrics...", file=sys.stderr)
        basic = self.basic_metrics()
        results.update({f'basic_{k}': v for k, v in basic.items()})
        
        # Average path length
        print("Computing average path length...", file=sys.stderr)
        avg_path = self.average_path_length()
        results['avg_path_length'] = avg_path
        
        # Degree distribution statistics
        print("Computing degree statistics...", file=sys.stderr)
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]
        results['in_degree_median'] = float(np.median(in_degrees))
        results['out_degree_median'] = float(np.median(out_degrees))
        
        # Fagiolo clustering
        clustering_results, clustering_avg = self.fagiolo_clustering_fast()
        results.update({f'clustering_{k}': v for k, v in clustering_avg.items()})
        
        # Small-worldness
        sw_metrics = self.compute_small_worldness(num_random)
        results.update({f'smallworld_{k}': v for k, v in sw_metrics.items()})
        
        # Create DataFrame and save
        df = pd.DataFrame([results])
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Results saved to: {output_path}", file=sys.stderr)
        
        # Print summary
        print("\n" + "="*70, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"Nodes: {self.n}, Edges: {self.m}", file=sys.stderr)
        print(f"Avg Path Length: {avg_path:.4f}", file=sys.stderr)
        print(f"Clustering (C^D): {clustering_avg['overall']:.4f}", file=sys.stderr)
        print(f"Small-worldness σ: {sw_metrics['sigma']:.4f}", file=sys.stderr)
        if sw_metrics['is_small_world']:
            print("✓ Small-world properties detected", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze directed graphs with Fagiolo clustering coefficients',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dependency_graph.dot results.csv
  %(prog)s graph.dot output.csv --random-iterations 20
  %(prog)s input.dot output.csv -r 5

Output CSV contains:
  - Basic graph metrics (nodes, edges, density, degrees)
  - Average path length
  - Fagiolo clustering coefficients (overall, cycle, middleman, in, out)
  - Small-worldness metrics (σ, comparison with random graphs)
        """
    )
    
    parser.add_argument('input', type=str, 
                       help='Input graph file in DOT format')
    parser.add_argument('output', type=str, 
                       help='Output CSV file path')
    parser.add_argument('-r', '--random-iterations', type=int, default=10,
                       help='Number of random graphs for small-world comparison (default: 10)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.suffix.lower() in ['.dot', '.gv']:
        print(f"Warning: Input file should be in DOT format (.dot or .gv)", file=sys.stderr)
    
    # Validate output path
    output_path = Path(args.output)
    if output_path.suffix.lower() != '.csv':
        print(f"Warning: Output file should have .csv extension", file=sys.stderr)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run analysis
        analyzer = FagioloClusteringAnalyzer(str(input_path))
        analyzer.analyze_and_export(str(output_path), args.random_iterations)
        
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()