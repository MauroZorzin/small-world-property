# Software Dependency Analysis Pipeline

A professional, modular pipeline for extracting, analyzing, and visualizing software dependency structures from code repositories.

## Overview

This pipeline consists of two main modules:

1. **Extraction Module** (`extract_dependencies.py`) - Extracts dependency graphs from source code
2. **Analysis Module** (`analyze_dependencies.py`) - Analyzes graphs and computes comprehensive metrics

Both modules can be run independently or together as a complete pipeline.

## Features

### Extraction Module
- ✅ Batch processing of multiple repositories
- ✅ Support for 7 programming languages (Java, Python, C++, Kotlin, Go, Ruby, POM)
- ✅ Robust error handling with detailed logging
- ✅ Progress tracking with progress bars
- ✅ Timeout protection per project
- ✅ Extraction report generation (JSON)

### Analysis Module
- ✅ Comprehensive graph metrics (30+ metrics per project)
- ✅ Separate analysis for direct and indirect dependencies
- ✅ Small-world properties (clustering, path length, sigma, omega)
- ✅ Connectivity metrics (connected components, weakly connected components)
- ✅ Centrality metrics (degree, betweenness)
- ✅ Multiple export formats (CSV, JSON, summary report)
- ✅ Configurable analysis depth

## Installation

### Prerequisites

1. **Python 3.8 or higher**
2. **Depends Tool** - Download from [Depends GitHub](https://github.com/multilang-depends/depends/releases)

### Setup

```bash
# Clone or navigate to the repository
cd small-world-property

# Install Python dependencies
pip install -r requirements.txt

# Extract the depends tool (if not already done)
# Ensure depends-0.9.7-package-20221104a/depends.exe exists
```

## Directory Structure

```
small-world-property/
├── extract_dependencies.py      # Extraction module
├── analyze_dependencies.py      # Analysis module
├── config.yaml                  # Configuration file (optional)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── depends-0.9.7-package-20221104a/  # Depends tool
│   └── depends.exe
│
├── repos/                       # INPUT: Source code repositories
│   ├── project1/
│   ├── project2/
│   └── ...
│
├── depends-out/                 # OUTPUT: .dot dependency graphs
│   ├── project1.dot
│   ├── project2.dot
│   ├── extraction_report.json  # Extraction statistics
│   └── ...
│
├── results/                     # OUTPUT: Analysis results
│   ├── dependency_metrics.csv  # Main results (CSV)
│   ├── dependency_metrics.json # Main results (JSON)
│   └── analysis_summary.txt    # Human-readable summary
│
└── logs/                        # OUTPUT: Execution logs
    ├── extraction_YYYYMMDD_HHMMSS.log
    └── analysis_YYYYMMDD_HHMMSS.log
```

## Usage

### Option 1: Run Complete Pipeline (Windows)

Double-click or run:
```bash
run_full_pipeline.bat
```

### Option 2: Run Modules Separately (Windows)

**Step 1: Extract dependencies**
```bash
run_extraction.bat
```

**Step 2: Analyze graphs**
```bash
run_analysis.bat
```

### Option 3: Command Line (Cross-platform)

#### Extract Dependencies

```bash
# Basic extraction
python extract_dependencies.py

# With custom configuration
python extract_dependencies.py --config config.yaml

# Specify language and folders
python extract_dependencies.py --repos ./my-repos --language python --output ./my-graphs

# With extraction report
python extract_dependencies.py --save-report

# Custom timeout (seconds)
python extract_dependencies.py --timeout 600
```

#### Analyze Dependencies

```bash
# Basic analysis
python analyze_dependencies.py

# With custom configuration
python analyze_dependencies.py --config config.yaml

# Specify input/output folders
python analyze_dependencies.py --input ./my-graphs --output ./my-results

# Skip indirect dependency analysis (faster)
python analyze_dependencies.py --no-indirect

# Skip small-world computation (much faster for large graphs)
python analyze_dependencies.py --no-small-world

# Export only CSV
python analyze_dependencies.py --csv-only
```

## Configuration

Create a `config.yaml` file to customize behavior:

```yaml
# Extraction configuration
depends_path: "depends-0.9.7-package-20221104a/depends.exe"
repos_folder: "repos"
output_folder: "depends-out"
language: "java"
extraction_timeout: 300

# Analysis configuration
dot_folder: "depends-out"
output_folder: "results"
log_folder: "logs"
compute_small_world: true
analyze_indirect: true
```

### Supported Languages

- `cpp` - C/C++
- `python` - Python
- `java` - Java
- `kotlin` - Kotlin
- `go` - Go
- `ruby` - Ruby
- `pom` - Maven POM files

## Output Files

### Extraction Outputs

1. **`.dot` files** (`depends-out/*.dot`)
   - GraphViz DOT format dependency graphs
   - One file per project
   - Used as input for analysis module

2. **Extraction report** (`depends-out/extraction_report.json`)
   - Statistics on extraction success/failure
   - List of failed projects with reasons
   - Generated when using `--save-report` flag

### Analysis Outputs

1. **CSV Results** (`results/dependency_metrics.csv`)
   - Main results file with all metrics
   - One row per project
   - Easily importable to Excel, R, Python, etc.

2. **JSON Results** (`results/dependency_metrics.json`)
   - Same data as CSV in JSON format
   - For programmatic access

3. **Summary Report** (`results/analysis_summary.txt`)
   - Human-readable summary
   - Aggregate statistics
   - Small-world property analysis

### Log Files

- **Extraction logs** (`logs/extraction_YYYYMMDD_HHMMSS.log`)
- **Analysis logs** (`logs/analysis_YYYYMMDD_HHMMSS.log`)

## Metrics Explained

### Basic Metrics
- `num_nodes` - Number of files/modules
- `num_edges` - Number of dependencies
- `density` - Graph density (0-1)

### Connectivity Metrics
- `is_weakly_connected` - Whether graph is connected
- `num_weakly_connected_components` - Number of disconnected components
- `percent_nodes_largest_wcc` - Percentage of nodes in largest component

### Centrality Metrics
- `avg_degree_centrality` - Average importance based on connections
- `avg_betweenness_centrality` - Average importance based on paths

### Small-World Metrics
- `clustering_coefficient` - Local clustering measure (0-1)
- `avg_path_length` - Average shortest path length
- `small_world_sigma` (σ) - Small-world index
  - σ > 1 indicates small-world properties
  - High clustering + short paths
- `small_world_omega` (ω) - Alternative small-world measure
  - ω ≈ 0 indicates small-world properties

### Direct vs Indirect Dependencies

- **Direct**: Explicitly declared in code (e.g., import statements)
- **Indirect**: Transitive dependencies (if A→B and B→C, then A→C indirectly)

All metrics are computed separately for both dependency types (prefixed with `direct_` or `indirect_`).

## Performance Considerations

### Extraction
- Average: 10-30 seconds per project
- Large projects: up to 5 minutes
- Use `--timeout` to limit per-project time

### Analysis
- Small graphs (<100 nodes): seconds
- Medium graphs (100-1000 nodes): 1-2 minutes
- Large graphs (>1000 nodes): 5-10 minutes
- Use `--no-small-world` to skip expensive computations

### Optimization Tips

1. **Skip small-world for large datasets**:
   ```bash
   python analyze_dependencies.py --no-small-world
   ```

2. **Skip indirect analysis**:
   ```bash
   python analyze_dependencies.py --no-indirect
   ```

3. **Process subsets**:
   - Move some `.dot` files to a different folder
   - Analyze in batches

## Troubleshooting

### Common Issues

**1. "pydot not found"**
```bash
pip install pydot pyparsing
```

**2. "depends.exe not found"**
- Extract `depends-0.9.7-package-20221104a.zip`
- Verify path in config.yaml
- Check Windows: ensure `.exe` extension

**3. "Empty .dot file"**
- Repository may not contain parseable code
- Check depends.log for errors
- Try different language setting

**4. "Graph is disconnected"**
- This is normal for some projects
- Analysis uses largest connected component
- Metrics computed on component only

**5. Extraction timeouts**
- Increase timeout: `--timeout 600`
- Large monorepos may need 10+ minutes

**6. Analysis memory errors**
- Skip small-world: `--no-small-world`
- Transitive closure is memory-intensive
- Process large graphs separately

### Debug Mode

Add `--verbose` flag or edit logging in source:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Examples

### Example 1: Quick Analysis of Java Projects

```bash
# Extract
python extract_dependencies.py --language java --save-report

# Analyze (fast mode)
python analyze_dependencies.py --no-small-world --csv-only
```

### Example 2: Complete Python Analysis

```bash
# Extract with longer timeout
python extract_dependencies.py --language python --timeout 600

# Full analysis
python analyze_dependencies.py
```

### Example 3: Analyze Existing .dot Files

```bash
# Skip extraction, analyze only
python analyze_dependencies.py --input ./existing-graphs
```

### Example 4: Custom Workflow

```bash
# Extract to custom location
python extract_dependencies.py --output ./graphs-v1

# Analyze from custom location
python analyze_dependencies.py --input ./graphs-v1 --output ./results-v1
```

## Extending the Pipeline

### Add Custom Metrics

Edit `analyze_dependencies.py`, add to `GraphAnalyzer` class:

```python
def compute_custom_metrics(self, graph: nx.Graph) -> Dict:
    return {
        "my_metric": my_calculation(graph),
        "another_metric": another_calculation(graph)
    }
```

Then call in `analyze_project()`:
```python
direct_metrics.update(self.compute_custom_metrics(direct_graph))
```

### Add Visualization

Install visualization libraries:
```bash
pip install matplotlib seaborn plotly networkx[default]
```

Create a new `visualizer.py` module to generate plots.

### Support New Languages

The Depends tool supports many languages. Just use the appropriate language code:

```bash
python extract_dependencies.py --language <language_code>
```

## Citation

If you use this pipeline in research, please cite:

```bibtex
@software{dependency_analysis_pipeline,
  title = {Software Dependency Analysis Pipeline},
  author = {[Your Name]},
  year = {2026},
  url = {[Your Repository URL]}
}
```

## References

- **Depends Tool**: https://github.com/multilang-depends/depends
- **NetworkX**: https://networkx.org/
- **Small-World Networks**: Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.
- **Small-World Sigma**: Humphries, M. D., & Gurney, K. (2008). Network 'small-world-ness': a quantitative method for determining canonical network equivalence. *PLoS ONE*, 3(4), e0002051.
- **Small-World Omega**: Telesford, Q. K., et al. (2011). The ubiquity of small-world networks. *Brain connectivity*, 1(5), 367-375.

## License

[Your License Here]

## Support

For issues or questions:
- Check logs in `logs/` folder
- Review failed projects in extraction report
- Open an issue on GitHub

---

**Version**: 2.0  
**Last Updated**: January 2026
