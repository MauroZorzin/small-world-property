# Small-World Property Analysis: Directed Graph Metrics & Architectural Smells

---

## How to Use the Code

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `networkx`, `openpyxl`

### Step 2: Run the Analysis Notebook

Open and execute `analyse.ipynb` in Jupyter:

```bash
jupyter notebook analyse.ipynb
```

Or use JupyterLab:

```bash
jupyter lab analyse.ipynb
```

---

## File Structure and Content

### Core Python Files

| File | Purpose |
|------|---------|
| `analyse.ipynb` | **Main analysis notebook** – Loads DV8 & FAG data, computes correlations, generates plots and statistics |
| `pipeline.py` | Standalone CLI tool for computing graph metrics on DOT files; generates FAG-style CSV output |
| `test_fagiolo_static.py` | Unit tests for Fagiolo clustering coefficient calculations |

### Input Directories

| Directory | Content |
|-----------|---------|
| `dv8-reports/` | DV8 analysis results for each project (anti-pattern costs in Excel format) |
| `results/` | FAG (Fagiolo Aggregation) CSV outputs for each project (graph metrics per project) |
| `depends-out-dot/` | Dependency graph files in GraphViz DOT format |

---

## CSV & PNG Files

| File | Content |
|------|---------|
| **`dv8_antipattern_summary.csv`** | Anti-pattern densities by project (Clique, PackageCycle, UnhealthyInheritance, Total_AntiPattern_Density) |
| **`fag_pipeline_summary.csv`** | Complete graph metrics for all projects (nodes, edges, clustering coefficients, path lengths, connected components) |
| **`merged_dv8_fag_summary.csv`** | Combined anti-patterns + graph metrics for all 10 projects (master dataset used for correlation analysis) |
| **`sigma_table_by_project.csv`** | Small-worldness sigma values organized by project (45 sigma columns: 5 types × 9 scopes) |
| **`sigma_long_form.csv`** | Long-form sigma table with three columns: `project_name`, `sigma_metric`, `sigma_value`, `smallworld_status` (True/False for sigma > 1) |
| **`smallworld_tendency_by_scope.csv`** | Fraction of projects exhibiting small-world properties by scope (Full→LSCC, Full→AllSCC, etc.) |
| **`smallworld_tendency_by_sigma_type.csv`** | Fraction showing small-world by sigma type (Overall, Cycle, Middleman, In, Out) |
| **`smallworld_tendency_by_sigma_metric.csv`** | Fraction showing small-world for each specific metric combination |
| **`coverage_lscc_allscc_summary.csv`** | Node and edge coverage percentages for LSCC and AllSCC relative to full graph |
| **`normality_results.csv`** | Shapiro-Wilk test results (variable, statistic, p-value, is_normal flag) for all metrics |
| **`pearson_results.csv`** | Pearson correlation coefficients and p-values between architectural metrics and quality metrics |
| **`spearman_results.csv`** | Spearman rank correlation coefficients and p-values (non-parametric alternative) |
| **`compare_path_methods.png`** | Scatter plots comparing three path-length methods (LSCC, AllSCC, Undirected) |
| **`lscc_vs_allscc_graph.png`** | Bar charts showing LSCC vs. AllSCC node/edge coverage |
| **`correlation_plots/positive/*.png`** | Individual scatter plots for each positive correlation (one per metric) |
| **`correlation_plots/negative/*.png`** | Individual scatter plots for each negative correlation (one per metric) |

---


### Projects Analyzed

The analysis includes 10 open-source Java projects:

| Project | Type |
|---------|------|
| **activemq** | Message broker |
| **archiva** | Repository manager |
| **depends** | Dependency analyzer |
| **druid** | OLAP data store |
| **geode** | Distributed cache | geode\geode-core\src\test folder has been removed do to limit in number of file analised
| **jackrabbit** | Content repository |
| **jena** | RDF/OWL framework |
| **karaf** | Application container |
| **phoenix** | SQL query engine |
| **solr** | Search platform |

---
