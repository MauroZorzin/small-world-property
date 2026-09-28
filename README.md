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
python generate_random_sample.py results/<project>/degree_sequence.json --out-dir results/<project>/ --converge-threshold 2.0
```
Generates degree-preserving random graphs and writes their metrics to a new file under
`results/<project>/random_samples/`. Safe to run again anytime to add more samples
without recomputing anything. Optional `--seed` for reproducibility.

Two mutually exclusive modes

| Parameter | Meaning |
|---|---|
| `--count N` | Generates exactly N graphs. If no mode is given the default is 1 |
| `--converge-threshold PCT` | Generates in batches until every clustering and path length metric has a relative standard error of the mean (SEM) under PCT percent. The check uses all accumulated samples, those of this run and those of every file already present in `random_samples/` |
| `--batch-size N` | Only with `--converge-threshold`. Samples per batch before rechecking convergence. Default 50 |
| `--max-samples N` | Only with `--converge-threshold`. Safety cap on the total accumulated samples. Default 5000 |

The null model is a configuration model that preserves each node's exact degree
rather than an Erdos-Renyi graph, which is the more conservative choice for
networks with heterogeneous degree distributions like software dependency graphs.

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
| `test_fagiolo_static.py` | Unit tests for `metrics_logic.fagiolo_on_graph` against hand-derived expected values |


### Output Files

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
| `correlation_results.csv` | Pearson and Spearman r and p for every architecture x quality pair (196 tests), with Benjamini-Hochberg (`_bh`) and Benjamini-Yekutieli (`_by`) FDR correction. `sigma_full_correlation_results.csv` (180 tests) and `sigma_matched_correlation_results.csv` (60 tests) have the same columns for their own families. BY is valid under any dependence between tests, BH assumes independence or positive dependence |
| `best_estimators_summary_all.csv` / `_sigma_all.csv` / `_sigma.csv` | Best-correlating estimator per quality metric for the all-variables, sigma (45) and sigma scope-matched (15) candidate pools, with BH and BY p-values (`*` = BH significant, `**` = also BY significant) |
| `best_estimators_candidates_all.csv` / `_sigma_all.csv` / `_sigma.csv` | Every candidate considered behind each summary above, ranked |
| `best_estimators_comparison.csv` | The three best-estimator summaries stacked in one table |
| `*_full_test_overview_bh.png` / `*_full_test_overview_by.png` | Sign and significance of every test and spread of r per quality metric, one figure per FDR method, for the three test families |
| `correlation_plots/bh/*.png`, `correlation_plots/by/*.png` | Scatter plot for each architecture/quality pair significant after BH, and for those that also survive BY |
| `coverage_lscc_allscc_summary.csv` / `coverage_lscc_allscc.png` | Node and edge coverage of LSCC and AllSCC relative to the full graph |

---

### Projects Analyzed

10 open-source Java projects:

| Project | Type | Repository | Branch | Commit |
|---|---|---|---|---|
| activemq | Message broker | [apache/activemq](https://github.com/apache/activemq) | `main` | [`ef789aad26`](https://github.com/apache/activemq/commit/ef789aad26) |
| archiva | Repository manager | [apache/archiva](https://github.com/apache/archiva) | `master` | [`2beaf86490`](https://github.com/apache/archiva/commit/2beaf86490) |
| depends | Dependency analyzer | [multilang-depends/depends](https://github.com/multilang-depends/depends) | `master` | [`bf4c41d03a`](https://github.com/multilang-depends/depends/commit/bf4c41d03a) |
| druid | JDBC connection pool / SQL monitor | [alibaba/druid](https://github.com/alibaba/druid) | `master` | [`78fa7415c4`](https://github.com/alibaba/druid/commit/78fa7415c4) |
| geode¹ ² | Distributed cache | [apache/geode](https://github.com/apache/geode) | `develop` | [`b0b2dab9de`](https://github.com/apache/geode/commit/b0b2dab9de) |
| jackrabbit | Content repository | [apache/jackrabbit](https://github.com/apache/jackrabbit) | `trunk` | [`bb1f7e3595`](https://github.com/apache/jackrabbit/commit/bb1f7e3595) |
| jena² | RDF/OWL framework | [apache/jena](https://github.com/apache/jena) | `main` | [`9479c0490a`](https://github.com/apache/jena/commit/9479c0490a) |
| karaf | Application container | [apache/karaf](https://github.com/apache/karaf) | `main` | [`41fb3f7228`](https://github.com/apache/karaf/commit/41fb3f7228) |
| phoenix | SQL query engine | [apache/phoenix](https://github.com/apache/phoenix) | `master` | [`93203b0812`](https://github.com/apache/phoenix/commit/93203b0812) |
| solr | Search platform | [apache/solr](https://github.com/apache/solr) | `main` | [`e92700f0f8`](https://github.com/apache/solr/commit/e92700f0f8) |

¹ `geode-core/src/test` was removed to reduce the file count analyzed.
² DV8 throws an exception when analyzing jena and geode; this does not affect the data used in this study.



