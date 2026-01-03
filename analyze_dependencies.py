"""
Dependency Graph Analysis Module
Analyzes .dot dependency graphs and computes comprehensive metrics
"""

import os
import sys
import csv
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import networkx as nx
from tqdm import tqdm
import yaml


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging with both file and console handlers"""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class GraphAnalyzer:
    """Analyzes dependency graphs and computes comprehensive metrics"""
    
    def __init__(self, logger: logging.Logger, compute_small_world: bool = True):
        self.logger = logger
        self.compute_small_world = compute_small_world
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "failed_files": []
        }
    
    def load_graph(self, dot_file: Path) -> Optional[nx.DiGraph]:
        """Load graph from DOT file and ensure it's a simple DiGraph"""
        try:
            # Load the graph
            graph = nx.nx_pydot.read_dot(str(dot_file))
            
            # Convert to simple DiGraph (remove parallel edges and self-loops)
            if isinstance(graph, nx.MultiDiGraph):
                self.logger.debug(f"Converting MultiDiGraph to DiGraph")
                simple_graph = nx.DiGraph()
                simple_graph.add_nodes_from(graph.nodes())
                for u, v in graph.edges():
                    if not simple_graph.has_edge(u, v):
                        simple_graph.add_edge(u, v)
                graph = simple_graph
            
            # Remove self-loops
            graph.remove_edges_from(nx.selfloop_edges(graph))
            
            self.logger.debug(f"Loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            return graph
        except Exception as e:
            self.logger.error(f"Error loading {dot_file.name}: {e}")
            return None
    
    def separate_dependencies(self, graph: nx.DiGraph) -> Tuple[Optional[nx.DiGraph], Optional[nx.DiGraph]]:
        """Separate direct and indirect (transitive) dependencies"""
        try:
            direct_graph = graph.copy()
            
            self.logger.debug("Computing transitive closure...")
            transitive_graph = nx.transitive_closure(graph)
            indirect_graph = nx.difference(transitive_graph, direct_graph)
            
            self.logger.debug(f"Direct: {direct_graph.number_of_edges()} edges, Indirect: {indirect_graph.number_of_edges()} edges")
            return direct_graph, indirect_graph
            
        except Exception as e:
            self.logger.error(f"Error separating dependencies: {e}")
            return None, None
    
    def compute_basic_metrics(self, graph: nx.Graph) -> Dict:
        """Compute basic graph metrics"""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": nx.density(graph) if num_nodes > 1 else 0.0,
        }
    
    def compute_connectivity_metrics(self, graph: nx.Graph) -> Dict:
        """Compute connectivity metrics (works for both directed and undirected)"""
        metrics = {}
        
        try:
            if graph.is_directed():
                # Directed graph - use weakly connected components
                metrics["is_weakly_connected"] = nx.is_weakly_connected(graph)
                metrics["num_weakly_connected_components"] = nx.number_weakly_connected_components(graph)
                
                if graph.number_of_nodes() > 0:
                    wccs = list(nx.weakly_connected_components(graph))
                    if wccs:
                        largest_wcc = max(wccs, key=len)
                        largest_subgraph = graph.subgraph(largest_wcc)
                        
                        metrics["nodes_largest_wcc"] = largest_subgraph.number_of_nodes()
                        metrics["edges_largest_wcc"] = largest_subgraph.number_of_edges()
                        metrics["percent_nodes_largest_wcc"] = (metrics["nodes_largest_wcc"] / graph.number_of_nodes()) * 100
                        
                        if graph.number_of_edges() > 0:
                            metrics["percent_edges_largest_wcc"] = (metrics["edges_largest_wcc"] / graph.number_of_edges()) * 100
                        else:
                            metrics["percent_edges_largest_wcc"] = 0.0
            else:
                # Undirected graph - use connected components
                metrics["is_connected"] = nx.is_connected(graph)
                metrics["num_connected_components"] = nx.number_connected_components(graph)
                
                if graph.number_of_nodes() > 0:
                    ccs = list(nx.connected_components(graph))
                    if ccs:
                        largest_cc = max(ccs, key=len)
                        largest_subgraph = graph.subgraph(largest_cc)
                        
                        metrics["nodes_largest_cc"] = largest_subgraph.number_of_nodes()
                        metrics["edges_largest_cc"] = largest_subgraph.number_of_edges()
                        metrics["percent_nodes_largest_cc"] = (metrics["nodes_largest_cc"] / graph.number_of_nodes()) * 100
                        
                        if graph.number_of_edges() > 0:
                            metrics["percent_edges_largest_cc"] = (metrics["edges_largest_cc"] / graph.number_of_edges()) * 100
                        else:
                            metrics["percent_edges_largest_cc"] = 0.0
        
        except Exception as e:
            self.logger.warning(f"Error computing connectivity: {e}")
        
        return metrics
    
    def compute_centrality_metrics(self, graph: nx.Graph) -> Dict:
        """Compute centrality metrics"""
        metrics = {}
        
        try:
            if graph.number_of_nodes() == 0:
                return metrics
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)
            if degree_centrality:
                metrics["avg_degree_centrality"] = sum(degree_centrality.values()) / len(degree_centrality)
                metrics["max_degree_centrality"] = max(degree_centrality.values())
            
            # Betweenness centrality (only for smaller graphs)
            if graph.number_of_nodes() < 500:
                betweenness = nx.betweenness_centrality(graph)
                if betweenness:
                    metrics["avg_betweenness_centrality"] = sum(betweenness.values()) / len(betweenness)
                    metrics["max_betweenness_centrality"] = max(betweenness.values())
        
        except Exception as e:
            self.logger.warning(f"Error computing centrality: {e}")
        
        return metrics
    
    def compute_small_world_metrics(self, graph: nx.Graph) -> Dict:
        """
        Compute small-world metrics on undirected graph
        MUST be called with an undirected, simple graph
        """
        metrics = {
            "clustering_coefficient": None,
            "avg_path_length": None,
            "small_world_sigma": None,
            "small_world_omega": None,
        }
        
        if not self.compute_small_world:
            return metrics
        
        try:
            # Ensure graph is undirected and simple
            if graph.is_directed():
                self.logger.debug("Converting to undirected")
                graph = graph.to_undirected()
            
            # Ensure simple graph (no multi-edges)
            if isinstance(graph, nx.MultiGraph):
                self.logger.debug("Converting MultiGraph to Graph")
                simple_graph = nx.Graph()
                simple_graph.add_nodes_from(graph.nodes())
                for u, v in graph.edges():
                    if not simple_graph.has_edge(u, v):
                        simple_graph.add_edge(u, v)
                graph = simple_graph
            
            if graph.number_of_nodes() < 3:
                self.logger.debug(f"Graph too small: {graph.number_of_nodes()} nodes")
                return metrics
            
            # Must be connected for path length calculation
            if not nx.is_connected(graph):
                self.logger.debug("Graph not connected, cannot compute small-world metrics")
                return metrics
            
            # Compute clustering coefficient
            self.logger.debug(f"Computing clustering ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
            clustering = nx.average_clustering(graph)
            metrics["clustering_coefficient"] = clustering
            
            # Compute average path length
            self.logger.debug("Computing average path length")
            avg_path = nx.average_shortest_path_length(graph)
            metrics["avg_path_length"] = avg_path
            
            self.logger.debug(f"C={clustering:.4f}, L={avg_path:.4f}")
            
            # Generate random comparison graph
            degree_sequence = [d for n, d in graph.degree()]
            
            if sum(degree_sequence) % 2 != 0:
                degree_sequence[0] += 1
            
            try:
                # Configuration model preserves degree distribution
                random_graph = nx.configuration_model(degree_sequence)
                random_graph = nx.Graph(random_graph)
                random_graph.remove_edges_from(nx.selfloop_edges(random_graph))
                
                # Get largest component
                if not nx.is_connected(random_graph):
                    largest_cc = max(nx.connected_components(random_graph), key=len)
                    random_graph = random_graph.subgraph(largest_cc).copy()
                
                random_clustering = nx.average_clustering(random_graph)
                random_path = nx.average_shortest_path_length(random_graph)
                
                # Small-world sigma
                if random_clustering > 0 and random_path > 0:
                    sigma = (clustering / random_clustering) / (avg_path / random_path)
                    metrics["small_world_sigma"] = sigma
                    self.logger.debug(f"Sigma={sigma:.4f}")
                
            except Exception as e:
                self.logger.debug(f"Could not compute sigma: {e}")
            
            # Small-world omega
            try:
                n_nodes = len(degree_sequence)
                avg_degree = sum(degree_sequence) / len(degree_sequence)
                k = max(2, min(int(avg_degree), n_nodes - 1))
                
                if k % 2 != 0:
                    k -= 1
                if k < 2:
                    k = 2
                
                lattice = nx.watts_strogatz_graph(n_nodes, k, 0)
                
                if nx.is_connected(lattice):
                    lattice_clustering = nx.average_clustering(lattice)
                    
                    if lattice_clustering > 0 and avg_path > 0:
                        omega = (random_path / avg_path) - (clustering / lattice_clustering)
                        metrics["small_world_omega"] = omega
                        self.logger.debug(f"Omega={omega:.4f}")
                        
            except Exception as e:
                self.logger.debug(f"Could not compute omega: {e}")
        
        except Exception as e:
            self.logger.error(f"Error computing small-world metrics: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return metrics
    
    def analyze_project(self, dot_file: Path, analyze_indirect: bool = True) -> Optional[Dict]:
        """
        Analyze a single project
        
        DIRECT: Uses largest weakly connected component (so metrics are computable)
        INDIRECT: Uses full graph as-is
        """
        project_name = dot_file.stem
        self.logger.info(f"Analyzing: {project_name}")
        
        # Load graph
        graph = self.load_graph(dot_file)
        if graph is None:
            self.stats["failed"] += 1
            self.stats["failed_files"].append({"project": project_name, "reason": "failed to load"})
            return None
        
        self.logger.info(f"  Loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Separate dependencies
        self.logger.debug("  Separating dependencies...")
        direct_graph, indirect_graph = self.separate_dependencies(graph)
        if direct_graph is None:
            self.stats["failed"] += 1
            self.stats["failed_files"].append({"project": project_name, "reason": "failed to separate"})
            return None
        
        result = {"project": project_name}
        
        # ===================================================================
        # ANALYZE DIRECT DEPENDENCIES (use largest WCC)
        # ===================================================================
        self.logger.info(f"  Analyzing DIRECT dependencies...")
        
        # Get largest weakly connected component
        if nx.is_weakly_connected(direct_graph):
            self.logger.debug("    Graph is weakly connected")
            direct_lcc = direct_graph.copy()
        else:
            largest_wcc = max(nx.weakly_connected_components(direct_graph), key=len)
            direct_lcc = direct_graph.subgraph(largest_wcc).copy()
            self.logger.info(f"    Using largest WCC: {direct_lcc.number_of_nodes()} nodes ({direct_lcc.number_of_nodes()/direct_graph.number_of_nodes()*100:.1f}%)")
        
        # Basic metrics on directed graph
        direct_metrics = self.compute_basic_metrics(direct_lcc)
        direct_metrics.update(self.compute_connectivity_metrics(direct_lcc))
        direct_metrics.update(self.compute_centrality_metrics(direct_lcc))
        
        # Convert to undirected for small-world metrics
        direct_undirected = direct_lcc.to_undirected()
        
        # Ensure simple graph
        if isinstance(direct_undirected, nx.MultiGraph):
            simple = nx.Graph()
            simple.add_nodes_from(direct_undirected.nodes())
            for u, v in direct_undirected.edges():
                if not simple.has_edge(u, v):
                    simple.add_edge(u, v)
            direct_undirected = simple
        
        # Compute small-world metrics
        self.logger.debug("    Computing small-world metrics...")
        sw_metrics = self.compute_small_world_metrics(direct_undirected)
        direct_metrics.update(sw_metrics)
        
        # Log results
        if sw_metrics.get("clustering_coefficient") is not None:
            self.logger.info(f"    [OK] Clustering: {sw_metrics['clustering_coefficient']:.4f}")
            self.logger.info(f"    [OK] Path length: {sw_metrics['avg_path_length']:.4f}")
            if sw_metrics.get("small_world_sigma"):
                self.logger.info(f"    [OK] Sigma: {sw_metrics['small_world_sigma']:.4f}")
        else:
            self.logger.warning(f"    [FAIL] Small-world metrics could not be computed")
        
        # Add to results with "direct_" prefix
        result.update({f"direct_{k}": v for k, v in direct_metrics.items()})
        
        # ===================================================================
        # ANALYZE INDIRECT DEPENDENCIES (use full graph)
        # ===================================================================
        if analyze_indirect and indirect_graph is not None:
            self.logger.info(f"  Analyzing INDIRECT dependencies...")
            self.logger.info(f"    Full graph: {indirect_graph.number_of_nodes()} nodes, {indirect_graph.number_of_edges()} edges")
            
            # Basic metrics on full directed graph
            indirect_metrics = self.compute_basic_metrics(indirect_graph)
            indirect_metrics.update(self.compute_connectivity_metrics(indirect_graph))
            indirect_metrics.update(self.compute_centrality_metrics(indirect_graph))
            
            # For small-world: convert to undirected and use largest CC
            # (Otherwise it would take forever on huge indirect graphs)
            indirect_undirected = indirect_graph.to_undirected()
            
            # Ensure simple graph
            if isinstance(indirect_undirected, nx.MultiGraph):
                self.logger.debug("    Converting to simple graph...")
                simple = nx.Graph()
                simple.add_nodes_from(indirect_undirected.nodes())
                for u, v in indirect_undirected.edges():
                    if not simple.has_edge(u, v):
                        simple.add_edge(u, v)
                indirect_undirected = simple
            
            # Use largest connected component for small-world
            if not nx.is_connected(indirect_undirected):
                self.logger.debug("    Using largest CC for small-world metrics...")
                largest_cc = max(nx.connected_components(indirect_undirected), key=len)
                indirect_for_sw = indirect_undirected.subgraph(largest_cc).copy()
                self.logger.info(f"    Largest CC: {indirect_for_sw.number_of_nodes()} nodes ({indirect_for_sw.number_of_nodes()/indirect_undirected.number_of_nodes()*100:.1f}%)")
            else:
                indirect_for_sw = indirect_undirected
            
            # Compute small-world (only if graph is reasonable size)
            if indirect_for_sw.number_of_nodes() < 10000:
                self.logger.debug("    Computing small-world metrics...")
                sw_metrics_indirect = self.compute_small_world_metrics(indirect_for_sw)
                indirect_metrics.update(sw_metrics_indirect)
                
                if sw_metrics_indirect.get("clustering_coefficient") is not None:
                    self.logger.info(f"    [OK] Clustering: {sw_metrics_indirect['clustering_coefficient']:.4f}")
            else:
                self.logger.info(f"    [SKIP] Graph too large ({indirect_for_sw.number_of_nodes()} nodes), skipping small-world")
                indirect_metrics.update({
                    "clustering_coefficient": None,
                    "avg_path_length": None,
                    "small_world_sigma": None,
                    "small_world_omega": None,
                })
            
            # Add to results with "indirect_" prefix
            result.update({f"indirect_{k}": v for k, v in indirect_metrics.items()})
        
        self.stats["successful"] += 1
        return result
    
    def analyze_all(self, dot_folder: Path, analyze_indirect: bool = True) -> List[Dict]:
        """Analyze all .dot files in folder"""
        if not dot_folder.exists():
            self.logger.error(f"Folder not found: {dot_folder}")
            return []
        
        dot_files = list(dot_folder.glob("*.dot"))
        
        if not dot_files:
            self.logger.warning(f"No .dot files found in: {dot_folder}")
            return []
        
        self.stats["total_files"] = len(dot_files)
        self.logger.info(f"Found {len(dot_files)} .dot files to analyze")
        
        results = []
        for dot_file in tqdm(dot_files, desc="Analyzing graphs"):
            result = self.analyze_project(dot_file, analyze_indirect)
            if result:
                results.append(result)
        
        return results
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Total files:           {self.stats['total_files']}")
        print(f"Successfully analyzed: {self.stats['successful']}")
        print(f"Failed:                {self.stats['failed']}")
        print(f"Success rate:          {(self.stats['successful'] / max(self.stats['total_files'], 1)) * 100:.1f}%")
        print("=" * 80)
        
        if self.stats["failed_files"]:
            print("\nFailed files:")
            for failed in self.stats["failed_files"]:
                print(f"  - {failed['project']}: {failed['reason']}")


class ResultsExporter:
    """Exports analysis results in multiple formats"""
    
    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    def export_csv(self, results: List[Dict], filename: str = "dependency_metrics.csv"):
        """Export results to CSV"""
        if not results:
            self.logger.warning("No results to export")
            return
        
        output_file = self.output_dir / filename
        
        try:
            fieldnames = list(results[0].keys())
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            self.logger.info(f"CSV exported: {output_file}")
        except Exception as e:
            self.logger.error(f"Error exporting CSV: {e}")
    
    def export_json(self, results: List[Dict], filename: str = "dependency_metrics.json"):
        """Export results to JSON"""
        if not results:
            self.logger.warning("No results to export")
            return
        
        output_file = self.output_dir / filename
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"JSON exported: {output_file}")
        except Exception as e:
            self.logger.error(f"Error exporting JSON: {e}")
    
    def export_summary(self, results: List[Dict], filename: str = "analysis_summary.txt"):
        """Export summary statistics"""
        if not results:
            return
        
        output_file = self.output_dir / filename
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DEPENDENCY ANALYSIS SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Total projects analyzed: {len(results)}\n")
                f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Direct metrics
                f.write("DIRECT DEPENDENCIES (Largest WCC)\n")
                d_nodes = [r.get('direct_num_nodes', 0) for r in results if r.get('direct_num_nodes') is not None]
                d_edges = [r.get('direct_num_edges', 0) for r in results if r.get('direct_num_edges') is not None]
                if d_nodes:
                    f.write(f"  Average nodes: {sum(d_nodes) / len(d_nodes):.2f}\n")
                    f.write(f"  Average edges: {sum(d_edges) / len(d_edges):.2f}\n")
                
                d_clustering = [r.get('direct_clustering_coefficient') for r in results if r.get('direct_clustering_coefficient') is not None]
                d_path = [r.get('direct_avg_path_length') for r in results if r.get('direct_avg_path_length') is not None]
                if d_clustering:
                    f.write(f"  Average clustering: {sum(d_clustering) / len(d_clustering):.4f}\n")
                if d_path:
                    f.write(f"  Average path length: {sum(d_path) / len(d_path):.4f}\n")
                
                d_sigma = [r.get('direct_small_world_sigma') for r in results if r.get('direct_small_world_sigma') is not None]
                if d_sigma:
                    f.write(f"  Average sigma: {sum(d_sigma) / len(d_sigma):.4f}\n")
                    f.write(f"  Projects with sigma > 1: {sum(1 for s in d_sigma if s > 1)}/{len(d_sigma)}\n\n")
                
                # Indirect metrics
                i_nodes = [r.get('indirect_num_nodes', 0) for r in results if r.get('indirect_num_nodes') is not None]
                if i_nodes:
                    f.write("INDIRECT DEPENDENCIES (Full Graph)\n")
                    i_edges = [r.get('indirect_num_edges', 0) for r in results if r.get('indirect_num_edges') is not None]
                    f.write(f"  Average nodes: {sum(i_nodes) / len(i_nodes):.2f}\n")
                    f.write(f"  Average edges: {sum(i_edges) / len(i_edges):.2f}\n")
            
            self.logger.info(f"Summary exported: {output_file}")
        except Exception as e:
            self.logger.error(f"Error exporting summary: {e}")


def load_config(config_file: Optional[Path] = None) -> dict:
    """Load configuration from YAML file or use defaults"""
    default_config = {
        "dot_folder": "depends-out",
        "output_folder": "results",
        "log_folder": "logs",
        "compute_small_world": True,
        "analyze_indirect": True
    }
    
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        except Exception as e:
            print(f"Warning: Error loading config file: {e}")
            print("Using default configuration")
    
    return default_config


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dependency graphs and compute metrics"
    )
    
    parser.add_argument("--config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("--input", type=Path, help="Input folder containing .dot files")
    parser.add_argument("--output", type=Path, help="Output folder for results")
    parser.add_argument("--no-indirect", action="store_true", help="Skip indirect dependency analysis")
    parser.add_argument("--no-small-world", action="store_true", help="Skip small-world metrics computation")
    parser.add_argument("--csv-only", action="store_true", help="Export only CSV")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.input:
        config["dot_folder"] = str(args.input)
    if args.output:
        config["output_folder"] = str(args.output)
    if args.no_indirect:
        config["analyze_indirect"] = False
    if args.no_small_world:
        config["compute_small_world"] = False
    
    # Setup paths
    dot_folder = Path(config["dot_folder"])
    output_folder = Path(config["output_folder"])
    log_folder = Path(config["log_folder"])
    
    # Setup logging
    logger = setup_logging(log_folder)
    logger.info("=" * 80)
    logger.info("Dependency Graph Analysis")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Input folder: {dot_folder}")
    logger.info(f"  Output folder: {output_folder}")
    logger.info(f"  Analyze indirect: {config['analyze_indirect']}")
    logger.info(f"  Compute small-world: {config['compute_small_world']}")
    
    # Create analyzer
    analyzer = GraphAnalyzer(
        logger=logger,
        compute_small_world=config["compute_small_world"]
    )
    
    # Analyze all graphs
    logger.info("\nStarting analysis...")
    results = analyzer.analyze_all(dot_folder, config["analyze_indirect"])
    
    if not results:
        logger.error("No results generated")
        analyzer.print_summary()
        sys.exit(1)
    
    # Print summary
    analyzer.print_summary()
    
    # Export results
    logger.info("\nExporting results...")
    exporter = ResultsExporter(output_folder, logger)
    
    exporter.export_csv(results)
    
    if not args.csv_only:
        exporter.export_json(results)
        exporter.export_summary(results)
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved to: {output_folder}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()