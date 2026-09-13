# Small-World Property Analysis: Directed Graph Metrics & Architectural Smells

Analyzes whether the file-level dependency graphs of open-source Java projects show
small-world network properties, and whether that correlates with architectural
anti-patterns reported by DV8.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Producing the Data

### 1. Extract the dependency graph (Depends)

For each project, extract the file-level dependency graph from the source tree with
[Depends](https://github.com/multilang-depends/depends):

```bash
depends.bat java <src_dir> <output_name> -g=file -s -f=json,dot --type-filter=Import,Call,Return,Throw,Implement,Extend,Create,Use,Cast,Annotation
```

Produces `<output_name>-file.dot` and `<output_name>-file.json`.

### 2. Extract anti-pattern costs (DV8)

Run a DV8 (ArchDia) architecture analysis on the same source tree. The resulting
`anti-pattern-cost.xlsx` report is read directly by the analysis notebook.

### 3. Compute the graph metrics

For each project, run the three scripts below in order.

```bash
python compute_real_metrics.py <output_name>-file.dot --out-dir results/<project>/
```
Loads the `.dot` file once and writes `real_metrics.csv` (exact graph metrics) and
`degree_sequence.json` (in/out degree sequence used by the next step).

```bash
python generate_random_sample.py results/<project>/degree_sequence.json --out-dir results/<project>/ --count 500
```
Generates `--count` degree-preserving random graphs and appends their metrics to a
new file under `results/<project>/random_samples/`. Safe to run again anytime to add
more samples without recomputing anything. Optional `--seed` for reproducibility.

```bash
python combine_results.py --project-dir results/<project>/
```
Reads `real_metrics.csv` and every file under `random_samples/`, computes the small
world sigma values, and writes `final_summary.csv`.

### 4. Run the analysis notebook

```bash
jupyter notebook analyse.ipynb
```
Reads `final_summary.csv` (per project, from step 3) and `anti-pattern-cost.xlsx`
(from step 2). All results are written to `analysis-results/`.

---

## File Structure and Content

### Core Python Files

| File | Purpose |
|---|---|
| `analyse.ipynb` | Main analysis notebook: loads DV8 and graph metrics, checks sigma collinearity, computes FDR-corrected correlations, builds the best-estimator summary, generates plots |
| `metrics_logic.py` | Shared metric computation (Fagiolo clustering, path lengths, SCC stats). No file I/O |
| `random_graph.py` | Shared random-graph generation (degree-preserving configuration model with repair). No file I/O |
| `compute_real_metrics.py` | CLI. Loads a `.dot` file once, writes `real_metrics.csv` and `degree_sequence.json` |
| `generate_random_sample.py` | CLI. Generates N random graphs, appends their metrics to a new file per invocation |
| `combine_results.py` | CLI. Combines `real_metrics.csv` and all random samples into `final_summary.csv` |
| `pipeline.py` | Legacy monolithic CLI, superseded by the three scripts above. Kept for reference |
| `test_fagiolo_static.py` | Unit tests for `metrics_logic.fagiolo_on_graph` against hand-derived expected values |

### Input Directories

| Directory | Content |
|---|---|
| `repos/` | Source checkouts of the 10 analyzed projects |
| `depends_9_10_2026_out/` | `.dot` and `.json` dependency graphs produced by Depends (step 1) |
| `dv8-out/` | DV8 project files and anti-pattern analysis results (step 2) |
| `results/<project>/` | Per-project pipeline output: `real_metrics.csv`, `degree_sequence.json`, `random_samples/*.csv`, `final_summary.csv` |

### Output Files (`analysis-results/`)

| File | Content |
|---|---|
| `dv8_antipattern_summary.csv` | Anti-pattern densities by project (Clique, PackageCycle, UnhealthyInheritance, Total_AntiPattern_Density) |
| `fag_pipeline_summary.csv` | Architectural metrics for all projects (nodes, edges, clustering, path lengths, sigma) |
| `merged_dv8_fag_summary.csv` | DV8 and architectural metrics merged, one row per project |
| `sample_convergence_summary.csv` | Per project: sample count, worst-converging metric, and whether all metrics are under the 2% relative-SEM threshold |
| `sample_convergence_detail.csv` | Relative standard error of the mean for every clustering/path-length metric, for every project |
| `sigma_table_by_project.csv` | The 45 sigma columns (5 patterns x 9 scopes) by project |
| `sigma_long_form.csv` | Long-form version of the sigma table, with `smallworld_status` |
| `sigma_correlation_heatmap.png` | Correlation between the 5 patterns, and between the 9 scopes |
| `sigma_boxplots.png` | Sigma value spread by pattern and by scope |
| `smallworld_tendency_by_scope.csv` / `_by_sigma_type.csv` / `_by_sigma_metric.csv` | Fraction of projects with sigma above 1, grouped different ways |
| `smallworld_tendency.png` | Bar charts of the two tendency tables above |
| `normality_results.csv` | Shapiro-Wilk normality test results per variable |
| `correlation_results.csv` | Pearson and Spearman r and p for every architecture x quality pair, with Benjamini-Hochberg FDR correction |
| `best_estimators_summary.csv` | Best-correlating estimator per quality metric, for the sigma-only and all-estimator candidate pools |
| `best_estimators_candidates.csv` | Every candidate considered behind the summary above, ranked |
| `correlation_plots/*.png` | Scatter plot for each FDR-significant architecture/quality pair |
| `coverage_lscc_allscc_summary.csv` / `coverage_lscc_allscc.png` | Node and edge coverage of LSCC and AllSCC relative to the full graph |

---

### Projects Analyzed

10 open-source Java projects:

| Project | Type | Commit |
|---|---|---|
| activemq | Message broker | [`ef789aad26`](https://github.com/apache/activemq/commit/ef789aad26) |
| archiva | Repository manager | [`2beaf86490`](https://github.com/apache/archiva/commit/2beaf86490) |
| depends | Dependency analyzer | [`bf4c41d03a`](https://github.com/multilang-depends/depends/commit/bf4c41d03a) |
| druid | JDBC connection pool / SQL monitor | [`78fa7415c4`](https://github.com/alibaba/druid/commit/78fa7415c4) |
| geode¹ | Distributed cache | [`b0b2dab9de`](https://github.com/apache/geode/commit/b0b2dab9de) |
| jackrabbit | Content repository | [`bb1f7e3595`](https://github.com/apache/jackrabbit/commit/bb1f7e3595) |
| jena | RDF/OWL framework | [`9479c0490a`](https://github.com/apache/jena/commit/9479c0490a) |
| karaf | Application container | [`41fb3f7228`](https://github.com/apache/karaf/commit/41fb3f7228) |
| phoenix | SQL query engine | [`93203b0812`](https://github.com/apache/phoenix/commit/93203b0812) |
| solr | Search platform | [`e92700f0f8`](https://github.com/apache/solr/commit/e92700f0f8) |

¹ `geode-core/src/test` was removed to reduce the file count analyzed.


