#!/usr/bin/env python3
"""
Directed Graph Analysis CLI Tool with Fagiolo Clustering
Usage: python graph_analyzer.py input.dot output.csv [--per-node output_nodes.csv] [--random-iterations N]
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
        
        # Store original node names (preserve file names)
        self.original_nodes = list(self.G.nodes())
        
        # Create node to index mapping for faster matrix operations
        self.node_to_idx = {node: idx for idx, node in enumerate(self.original_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Pre-compute adjacency matrix for performance
        self.A = nx.to_scipy_sparse_array(self.G, nodelist=self.original_nodes, 
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
            'out': c_out.tolist(),
            # Also store degree information
            'in_degree': d_in.tolist(),
            'out_degree': d_out.tolist(),
            'total_degree': d_tot.tolist(),
            'bilateral_degree': d_bilateral.tolist()
        }
        
        # Compute averages (ignoring inf and nan)
        averages = {}
        for key in ['overall', 'cycle', 'middleman', 'in', 'out']:
            valid_values = [v for v in results[key] if np.isfinite(v)]
            averages[key] = np.mean(valid_values) if valid_values else 0.0
        
        return results, averages

    def _parse_node_names_from_comments(self, dot_path: str):
        """
        Custom parser to extract node names from // ID:Name style comments.
        """
        id_to_name = {}
        # Regex to match // 4398:D:\Path\To\File.java
        comment_pattern = re.compile(r"^\s*//\s*(\d+):(.+)$")
        
        try:
            with open(dot_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = comment_pattern.match(line)
                    if match:
                        node_id = match.group(1)
                        # Clean up path to get just the filename or keep full path
                        full_path = match.group(2).strip()
                        # Optional: simplify name to just the class name
                        # simplified_name = full_path.split('\\')[-1] 
                        id_to_name[node_id] = full_path
        except Exception as e:
            print(f"Warning: Could not parse comments for node names: {e}", file=sys.stderr)
            
        return id_to_name

    def export_per_node_metrics(self, clustering_results: Dict, output_path: str, dot_path: str):
        """
        Modified export to include names parsed from DOT comments.
        """
        # 1. Parse names from comments
        id_to_name_map = self._parse_node_names_from_comments(dot_path)
        
        data = []
        for idx, node in enumerate(self.original_nodes):
            node_id_str = str(node)
            
            # 2. Lookup name from our map, fallback to ID if not found
            node_name = id_to_name_map.get(node_id_str, "Unknown")
            
            node_data = {
                'node_id': node_id_str,
                'node_name': node_name,
                'in_degree': int(clustering_results['in_degree'][idx]),
                'out_degree': int(clustering_results['out_degree'][idx]),
                'total_degree': int(clustering_results['total_degree'][idx]),
                'bilateral_degree': int(clustering_results['bilateral_degree'][idx]),
                'clustering_overall': clustering_results['overall'][idx],
                'clustering_cycle': clustering_results['cycle'][idx],
                'clustering_middleman': clustering_results['middleman'][idx],
                'clustering_in': clustering_results['in'][idx],
                'clustering_out': clustering_results['out'][idx]
            }
            data.append(node_data)
        
        df = pd.DataFrame(data)
        
        # Sort by total_degree descending (most connected nodes first)
        df = df.sort_values('total_degree', ascending=False)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved {len(df)} node records", file=sys.stderr)
        
        return df
    
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
            analyzer_random.original_nodes = list(G_random.nodes())
            analyzer_random.node_to_idx = {node: idx for idx, node in enumerate(analyzer_random.original_nodes)}
            analyzer_random.A = nx.to_scipy_sparse_array(G_random, 
                                                         nodelist=analyzer_random.original_nodes, 
                                                         format='csr', dtype=np.float32)
            
            _, C_random_avg = analyzer_random.fagiolo_clustering_fast()
            C_random_list.append(C_random_avg['overall'])
            L_random_list.append(analyzer_random.average_path_length())
        
        print("", file=sys.stderr)  # New line after progress
        
        # Filter out infinite values for meaningful statistics
        C_random_finite = [c for c in C_random_list if np.isfinite(c)]
        L_random_finite = [l for l in L_random_list if np.isfinite(l)]
        
        # Compute means and stds only from finite values
        C_random = np.mean(C_random_finite) if C_random_finite else 0.0
        L_random = np.mean(L_random_finite) if L_random_finite else float('inf')
        C_random_std = np.std(C_random_finite) if len(C_random_finite) > 1 else 0.0
        L_random_std = np.std(L_random_finite) if len(L_random_finite) > 1 else 0.0
        
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
            'C_random_std': C_random_std,
            'L_random_std': L_random_std,
            'is_small_world': sigma > 1 if np.isfinite(sigma) else False
        }
    
    def analyze_and_export(self, output_path: str, dot_path: str, per_node_path: Optional[str] = None, 
                          num_random: int = 10):
        """
        Run complete analysis and export results to CSV.
        
        Args:
            output_path: Path for summary CSV file
            per_node_path: Optional path for per-node metrics CSV
            num_random: Number of random graphs for small-world comparison
        """
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
        
        # Export per-node metrics if requested
        if per_node_path:
            self.export_per_node_metrics(clustering_results, per_node_path, dot_path )
        
        # Small-worldness
        sw_metrics = self.compute_small_worldness(num_random)
        results.update({f'smallworld_{k}': v for k, v in sw_metrics.items()})
        
        # Create DataFrame and save summary
        df = pd.DataFrame([results])
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Summary results saved to: {output_path}", file=sys.stderr)
        if per_node_path:
            print(f"✓ Per-node metrics saved to: {per_node_path}", file=sys.stderr)
        
        # Print summary
        print("\n" + "="*70, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"Nodes: {self.n}, Edges: {self.m}", file=sys.stderr)
        print(f"Avg Path Length: {avg_path:.4f}", file=sys.stderr)
        print(f"Clustering (C^D): {clustering_avg['overall']:.4f}", file=sys.stderr)
        print(f"  - Cycle (C^cyc): {clustering_avg['cycle']:.4f}", file=sys.stderr)
        print(f"  - Middleman (C^mid): {clustering_avg['middleman']:.4f}", file=sys.stderr)
        print(f"  - In (C^in): {clustering_avg['in']:.4f}", file=sys.stderr)
        print(f"  - Out (C^out): {clustering_avg['out']:.4f}", file=sys.stderr)
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
  %(prog)s graph.dot summary.csv --per-node nodes.csv
  %(prog)s input.dot output.csv --per-node per_node.csv --random-iterations 20
  %(prog)s graph.dot output.csv -p nodes.csv -r 5

Output Files:
  1. Summary CSV: Single row with aggregate metrics
  2. Per-Node CSV (optional): One row per node with clustering metrics

The per-node CSV preserves original node names (file names from .dot file).
        """
    )
    
    parser.add_argument('input', type=str, 
                       help='Input graph file in DOT format')
    parser.add_argument('output', type=str, 
                       help='Output CSV file path (summary metrics)')
    parser.add_argument('-p', '--per-node', type=str, default=None,
                       help='Output CSV file for per-node metrics (optional)')
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
    
    # Validate output paths
    output_path = Path(args.output)
    if output_path.suffix.lower() != '.csv':
        print(f"Warning: Output file should have .csv extension", file=sys.stderr)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    per_node_path = None
    if args.per_node:
        per_node_path = Path(args.per_node)
        if per_node_path.suffix.lower() != '.csv':
            print(f"Warning: Per-node output file should have .csv extension", file=sys.stderr)
        per_node_path.parent.mkdir(parents=True, exist_ok=True)
        per_node_path = str(per_node_path)
    
    try:
        # Run analysis
        analyzer = FagioloClusteringAnalyzer(str(input_path))
        analyzer.analyze_and_export(str(output_path),str(input_path), per_node_path, args.random_iterations)
        
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()