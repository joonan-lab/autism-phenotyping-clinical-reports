# autism-phenotyping-clinical-reports

Code for *"A language-model pipeline for multidimensional autism phenotyping from Korean clinical reports."*

The study uses 489 Korean clinical reports with structured narratives from 372 individuals at a single center. The reports span autism diagnostic and psychological assessment documentation. The ASD subset contains 346 reports from 251 individuals. Clinical text and patient-level analysis inputs are excluded from this repository.


## Study overview

A local Llama-3.1-8B model assigns all eligible sentences to one or more of 19 clinical domains. Five samples are drawn per sentence. A domain qualifies when it appears in at least three samples and has a mean confidence of at least 0.5 across the samples assigning that domain, with at most three domains per sentence. If no domain qualifies, the sentence is assigned to other/general. A psychiatrist independently annotated all sentences in 26 reports to provide gold labels.

Each eligible sentence contributes one unit of mass, divided equally among its assigned domains. Aggregating these contributions across all eligible sentences produces a 19-domain frequency vector for each report. The primary phenotype representation uses all eligible sentences.

Within the ASD cohort, domain proportions are transformed to log-ratio coordinates and adjusted for report type. Analyses assess dimensionality, mixture fit and partition reproducibility, and prototype distances. A two-layer fingerprint combines distance from the cohort centroid with standardized domain deviations. A separate ASD classifier supplies domain logit contributions, E, by allocating attention-weighted sentence logit scores across their assigned domains.

The ontology contains seven Core ASD domains, nine Associated and co-occurring features, and three Report elements and other domains. Codes and definitions are in `ontology/domains_19.yaml`.

## Code status

The R scripts in `figures/` and `tables/` use the current manuscript numbering and read processed inputs from external `figure_source` and `table_source` directories. The active scripts cover Figures 2–5, Figures S1–S7, Tables 1–2, Tables S1–S10, and Online Resource 2 (`ESM_2.xlsx`). Figure 1 is a schematic and has no R generation script here. Figure S3 contains the report-length matching analysis.

The Python pipeline constructs unweighted all-sentence vectors, fits the phenotype space, and calculates classifier-derived domain logit contributions. Analysis code reads private inputs from `REPORT_LLM_DATA_ROOT` and writes numerical results outside this checkout. Current figures and tables are generated separately from those results. Classifier training uses individual-disjoint outer folds and an individual-disjoint inner validation set for early stopping.

## Repository structure

| Path | Contents |
| --- | --- |
| `ontology/` | Nineteen-domain ontology and taxonomy builder |
| `pipeline/1_classifier/` | Classifier preprocessing, inner-validation training and sentence attention |
| `pipeline/2_silver_labeling/` | Local LLM labeling and gold-label utilities |
| `pipeline/3_multidomain_vectors/` | Equal-mass domain proportions from all eligible sentences |
| `pipeline/4_phenotype_space/` | Dimensionality, observed density models, reproducibility and fingerprints |
| `pipeline/5_domain_contributions/` | Sentence scores, attention-weighted contributions and domain E |
| `validation/silver_labels/` | Gold agreement, five-vote reliability, confidence calibration and domain sensitivity |
| `validation/classifier/` | Matched TF-IDF baselines and classifier calibration |
| `validation/domain_contributions/` | Domain deletion, held-out incremental prediction and gold allocation |
| `validation/sentence_selection/` | Full-sentence and attention top-10 profile agreement with full-report gold |
| `validation/report_length/` | A-type report-length matching |
| `figures/`, `tables/` | Main and supplementary R scripts |
| `analysis_paths.py` | External data and output paths |

## Requirements

Classifier training uses a GPU environment described in `pipeline/1_classifier/report_llm.yml`. Silver labeling requires the packages in `pipeline/2_silver_labeling/requirements.txt` and a locally served Ollama model. The Python vector and phenotype-space scripts use NumPy, SciPy, pandas, scikit-learn, matplotlib, and PyYAML. Sentence extraction also requires PyTorch and transformers.

The R scripts use `ggplot2`, `patchwork`, `jsonlite`, `openxlsx`, `ragg`, `reshape2`, `MASS`, `sn`, `xml2`, and `zip`. Figure export uses Arial, `ragg` for PNG, and a Cairo PDF device. Arial and Cairo support must be available to preserve the intended typography.

```bash
conda env create -f pipeline/1_classifier/report_llm.yml
conda activate report_llm
```

Reported training settings and software versions are documented in the manuscript and Supplementary Methods. The environment files and dependencies should be checked against those settings before rerunning an analysis.

## Generating the current figures and tables

Run each R script from its own directory so that `source("_common.R")` loads the matching helper. Use a consistent all-sentence analysis version for the figure and table inputs.

### Input directories

Set `REPORT_LLM_FIGURE_SOURCE_DIR` and `REPORT_LLM_TABLE_SOURCE_DIR` to the external directories containing the processed results. Their defaults are `~/Dropbox/ReportLLM_analysis/figure_source` and `~/Dropbox/ReportLLM_analysis/table_source`. These directories must exist. The supplementary-table helper reads `domain_vectors_meta_latest.json` when it is sourced.

Use the following filenames in `figure_source`.

| Script | Inputs |
| --- | --- |
| `Figure2.R` | `per_domain_all_sentences.tsv` |
| `Figure3.R` | `domain_vectors_proportion_latest.tsv`, `domain_vectors_meta_latest.json`, `step11_ados_variance_share_results.json` |
| `Figure4.R` | `observed_bic.tsv`, `step5_seed_stability_results.json`, `bootstrap_permutation_results.json`, `step4_residual_tail_results.json`, `domain_vectors_proportion_latest.tsv`, `domain_vectors_meta_latest.json`, `controlled_individual_profiles.tsv` |
| `Figure5.R` | `domain_vectors_proportion_latest.tsv`, `domain_vectors_meta_latest.json`, `step9_variance_results.json`, `two_channel_fingerprints.tsv`, `exemplar_domains.tsv`, `exemplar_selection.json` |
| `FigureS1.R` | `calibration_A2_reliability_bins.tsv`, `calibration_A2_results_summary.json` |
| `FigureS2.R` | `t2_report_type_figure_data.json` |
| `FigureS3.R` | `length_matched_replicates.tsv`, `length_matched_summary.tsv` |
| `FigureS4.R` | `classifier_oof_predictions.tsv`, `classifier_model_audit_results.json` |
| `FigureS5.R` | `domain_associations.tsv`, `heldout_model_comparisons.tsv`, `gold_E_preservation.tsv` |
| `FigureS6.R` | `atype_only_mode_stability.json` |
| `FigureS7.R` | `domain_vectors_proportion_latest.tsv`, `domain_vectors_meta_latest.json`, `exemplar_selection.json` |

Use the following filenames in `table_source`. Supplementary tables and `ESM_2.R` also require `domain_vectors_meta_latest.json` for domain order.

| Script | Inputs |
| --- | --- |
| `Table1.R` | `metadata_489reports.csv` |
| `Table2.R` | Ontology definitions are embedded in the script |
| `TableS1.R` | `profile_baselines.json`, `profile_spearman_round12.json` |
| `TableS2.R` | `silver_labels_489reports.jsonl`, `gold_labels_26reports.jsonl`, `per_domain_all_sentences.tsv` |
| `TableS3.R` | `gold_sentence_selection_summary.tsv` |
| `TableS4.R` | `observed_bic.tsv` |
| `TableS5.R` | `step5_seed_stability_results.json`, `bootstrap_permutation_results.json` |
| `TableS6.R` | `step2_pseudocount_sensitivity_results.json` |
| `TableS7.R` | `baseline_metrics.tsv`, `paired_model_differences.tsv` |
| `TableS8.R` | `per_fold_metrics.tsv`, `fold_quality.tsv` |
| `TableS9.R` | `domain_associations.tsv` |
| `TableS10.R` | `sensitivity_drop_domains_results.json`, `sensitivity_downweight_results.json`, `sensitivity_heavytail_results.json`, `sensitivity_fingerprint_results.json` |
| `ESM_2.R` | `controlled_individual_profiles.tsv`, `two_channel_fingerprints.tsv`, `fold_quality.tsv`, `internal/ESM_2_report_id_mapping.tsv` |

Table S3 compares silver profiles constructed from all eligible sentences or up to ten attention-selected sentences against the same full-report gold profiles. It reports mean cosine similarity, total variation distance, and within-report domain-rank Spearman correlation. `validation/sentence_selection/gold_agreement.py` calculates the six summary rows from private sentence annotations, stored out-of-fold attention and eligible-sentence masks. It runs independently of the contribution and phenotype-space analyses.

Place `gold_sentence_selection_summary.tsv` in `REPORT_LLM_TABLE_SOURCE_DIR`, or set `REPORT_LLM_GOLD_SELECTION_SUMMARY` to its full path. The columns used by `TableS3.R` are `method`, `metric`, `mean`, `percentile_interval`, and `n_reports`. Methods are `full` and `attention_top10`. Metrics are `cosine_full_gold`, `TV_full_gold`, and `rank_correlation_full_gold`. Each `percentile_interval` cell is a JSON array containing the lower and upper limits. Clinical inputs and generated summaries remain outside this repository.

### Calculating the Table S3 input

`gold_agreement.py` uses the same fixed v4.1 out-of-fold classifier inputs as the reported comparison. It does not train a model or call a language model. Its private input contract is as follows.

| Input | Required contents |
| --- | --- |
| Classifier intermediate directory | `report_id_array.npy` and `patient_id_array.npy`, each in report order; `valid_sentence_mask_np.npy` with shape reports × padded sentences; `attention_matrices_np.npy` with shape reports × padded sentences × padded sentences |
| Silver JSONL | `report_id`, `sentence_idx`, `sentence`, and `labels`; each label has `domain_id`; every eligible sentence in each gold report must be represented |
| Gold JSONL | `report_id`, `sentence_idx`, `sentence`, and `labels` as a list of domain IDs; the complete annotated sentence set for each gold report |
| Domain metadata JSON | `domain_columns`, listing the same nineteen domains in their analysis order |

Sentence indices refer to classifier positions. Gold indices are not assumed to match silver indices. The script matches exact stripped sentence text within each report, checks repeated-text multiplicities, and rejects repeated texts with conflicting gold labels. Eligible-query/key attention is averaged across queries, and the ten highest weights are selected with stable ties in source sentence order. A report with fewer than ten eligible sentences contributes all of them.

Each selected sentence contributes mass 1, split equally among its assigned domains. The full and selected silver profiles are compared with the same complete gold profile using cosine, total variation distance and within-report domain-rank Spearman correlation. Means are over reports. The percentile intervals use 1,000 bootstrap samples of individuals, retaining all reports from a sampled individual, with seed 42. The reported gold comparison has 26 reports from 26 individuals. These intervals condition on the fixed classifier and annotations; they do not include model retraining or annotation uncertainty.

The default inputs follow `analysis_paths.py`. All four paths can be overridden explicitly. For the reported Table S3, run this before `TableS3.R` and use the fixed v4.1 inputs.

```bash
python validation/sentence_selection/gold_agreement.py \
  --intermediates /absolute/path/to/v4_1_oof_intermediates \
  --silver /absolute/path/to/silver_labels_489reports.jsonl \
  --gold /absolute/path/to/gold_labels_26reports.jsonl \
  --domain-meta /absolute/path/to/domain_vectors_meta_latest.json \
  --output "$REPORT_LLM_TABLE_SOURCE_DIR/gold_sentence_selection_summary.tsv"
```

Without `--output`, the summary is written to `$REPORT_LLM_OUTPUT_ROOT/validation/sentence_selection/gold_sentence_selection_summary.tsv`. Set `REPORT_LLM_GOLD_SELECTION_SUMMARY` to that file when using the default. The script rejects output paths inside this checkout and writes only the aggregate summary, without report IDs, individual IDs, sentence text or per-report arrays.

The internal ESM_2 mapping pairs `report_id` with `submission_id`. It remains with the private inputs. `ESM_2.R` uses original IDs for matching and exports RS-001 through RS-346. The output contains one sheet with 346 ASD report rows and 48 columns, including phenotype deviations and domain logit contributions.

### Output directories and example commands

The figure helpers write `PNG/` and `PDF/` subdirectories beneath the selected output directory. Table scripts write one XLSX per table. Their defaults use the parent of the script directory, so set output variables explicitly to keep generated files outside this checkout.

| Variable | Output |
| --- | --- |
| `REPORT_LLM_FIGURE_OUT_DIR` | Main figure PNG/PDF directories |
| `REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR` | Supplementary figure PNG/PDF directories |
| `REPORT_LLM_TABLE_OUT_DIR` | `Table1.xlsx` and `Table2.xlsx` |
| `REPORT_LLM_SUPPLEMENTARY_TABLE_OUT_DIR` | `TableS1.xlsx`–`TableS10.xlsx` and `ESM_2.xlsx` |

Run these examples from the repository root after replacing the two input paths.

```bash
export REPORT_LLM_FIGURE_SOURCE_DIR=/absolute/path/to/current/figure_source
export REPORT_LLM_TABLE_SOURCE_DIR=/absolute/path/to/current/table_source

output_root="$(mktemp -d)"
export REPORT_LLM_FIGURE_OUT_DIR="$output_root/figures/main"
export REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR="$output_root/figures/supplementary"
export REPORT_LLM_TABLE_OUT_DIR="$output_root/tables/main"
export REPORT_LLM_SUPPLEMENTARY_TABLE_OUT_DIR="$output_root/tables/supplementary"

(cd figures && for script in Figure[2-5].R; do Rscript "$script" || exit; done)
(cd figures/supplementary && for script in FigureS[1-7].R; do Rscript "$script" || exit; done)
(cd tables && for script in Table[1-2].R; do Rscript "$script" || exit; done)
(cd tables/supplementary && for script in TableS{1..10}.R; do Rscript "$script" || exit; done)
(cd tables/supplementary && Rscript ESM_2.R)
```

`ESM_2.R` also accepts a complete output filename as its single optional argument. Without that argument, it writes `ESM_2.xlsx` in `REPORT_LLM_SUPPLEMENTARY_TABLE_OUT_DIR`. It does not use a separate supplementary-data output variable.

## Python analysis order

The scripts in `pipeline/4_phenotype_space/` are numbered by analysis order. `run_all.sh` runs steps 1–11 after the report-type adjustment check. Step 12 uses the domain contributions calculated in stage 5.

| Script | Analysis |
| --- | --- |
| `step1_effective_dim.py` | Effective dimensionality |
| `step2_pseudocount_sensitivity.py` | Pseudocount sensitivity |
| `step3_gmm_vs_heavytail.py` | Observed mixture and unimodal fits |
| `step4_residual_tail.py` | Full-space and subspace tails |
| `step5_seed_stability.py` | Cross-seed partition reproducibility |
| `step6_bootstrap_permutation.py` | Bootstrap and label-permutation reproducibility |
| `step7_normative.py` | Prototype distance and domain deviations |
| `step8_domain_deviation_prevalence.py` | Prevalence of large domain deviations |
| `step9_variance.py` | Within- and between-group variance |
| `step10_atype_only_mode_stability.py` | A-type-only partition reproducibility |
| `step11_ados_variance_share.py` | CLR variance by domain group |
| `step12_exemplar.py` | Author-selected report illustration |


`REPORT_LLM_DATA_ROOT` defaults to `~/Dropbox/ReportLLM_analysis`. `REPORT_LLM_OUTPUT_ROOT` defaults to its `02_results` directory. Set either variable before running when using other directories. Classifier intermediates default to `pipeline/1_classifier/intermediates/asd_classifier` under the external data root and can be overridden with `CLASSIFIER_INTERMEDIATES_DIR`.

Required private inputs include eligible-sentence masks, report/individual IDs, labels, stored OOF predictions and fold splits, silver and gold annotations, and classifier states. Contribution extraction replays the saved classifier heads from `sentence_cls.npy` (report × sentence × hidden size) or `lhs_embs_np.npy` (report × sentence × token × hidden size). It uses the first token state of each eligible sentence. `extract_sentence_cls.py` reads the large token-state array by memory mapping and saves only those states without changing the source file. The optional `--from-cache` flag reuses stored full-precision sentence scores, attention weights and compatibility scores. These cached quantities support domain aggregation and deletion analyses without the token-state file. The contribution analyses use saved out-of-fold classifiers without retraining them.

```bash
export REPORT_LLM_DATA_ROOT="$HOME/Dropbox/ReportLLM_analysis"

# Classifier training, when recomputing the diagnostic model.
bash pipeline/1_classifier/fine_tuning/run_train_489reports.sh

# These analyses also run from existing private inputs and stored OOF predictions.
python pipeline/3_multidomain_vectors/build_domain_vectors.py
python validation/silver_labels/sentence_level.py
python validation/silver_labels/report_level.py
python validation/silver_labels/inter_vote_agreement.py
python validation/silver_labels/confidence_calibration.py
python validation/sentence_selection/gold_agreement.py
bash pipeline/4_phenotype_space/run_all.sh

# When preparing CLS states from token states, run once:
# python pipeline/5_domain_contributions/extract_sentence_cls.py --input /absolute/path/to/lhs_embs_np.npy
python pipeline/5_domain_contributions/extract_sentence_contributions.py
python pipeline/5_domain_contributions/aggregate_domain_contributions.py
python pipeline/5_domain_contributions/build_fingerprints.py
python pipeline/4_phenotype_space/step12_exemplar.py

python validation/domain_contributions/domain_deletion.py
python validation/domain_contributions/incremental_information.py
python validation/domain_contributions/gold_allocation.py
python validation/report_length/length_matching.py
python validation/classifier/tfidf_baselines.py
python validation/classifier/performance_calibration.py

for script in drop_domains_robustness downweight_robustness heavytail_sensitivity fingerprint_sensitivity; do
  python "validation/silver_labels/domain_set/$script.py"
done
```

GMM and resampling analyses can take substantial time. The phenotype runner uses the all-sentence vectors; it does not run sentence recombination, attention-based sentence selection, or separate A-only/P-only classifiers. Label permutation in the reproducibility analysis and size-matched random deletion in the E analysis remain part of the reported methods.

The selected example is an author-selected illustration. `step12_exemplar.py` locates RS-346 using the private report-ID mapping in `table_source/internal/ESM_2_report_id_mapping.tsv`. It does not apply the earlier relative-direction selection rules.

Copy numerical outputs from a consistent all-sentence analysis version into `figure_source` and `table_source`, using the filenames listed above. The existing table-source label files and metadata remain private inputs. The legacy filename `two_channel_fingerprints.tsv` is retained for the R input contract; the current generator writes phenotype deviations and signed domain E without relative-direction scores.

### Output folders

Under `REPORT_LLM_OUTPUT_ROOT`, `domain_vectors/` contains full-precision domain proportions, `phenotype_space/` contains representation analyses and the selected example, and `domain_contributions/` contains sentence/domain contributions. The `validation/` subdirectories contain the corresponding reported analysis results. Figure and table input collections remain in `figure_source/` and `table_source/` under the external data root.

## Data

The clinical reports contain potentially identifying and sensitive individual information and are not publicly available. Requests for access to de-identified supporting data are subject to approval by the Institutional Review Board of Kyung Hee University Hospital and a data use agreement. Contact the corresponding author about access conditions.

The configured data directories, collected figure/table inputs, model weights, and generated outputs are excluded from version control. Running analyses requires the relevant private inputs. Neither clinical text nor the internal report-ID mapping belongs in this repository.

`silver_labels_489reports.jsonl` contains the model-assigned labels for all 32,317 eligible sentences. `gold_labels_26reports.jsonl` contains the gold annotations for the 26-report validation set.

## Citation

```bibtex
@unpublished{Ryu_reportllm,
  title   = {A language-model pipeline for multidimensional autism phenotyping from Korean clinical reports},
  author  = {Ryu, Yeojin and An, Joon-Yong and Yoo, Hee Jeong and Bzdok, Danilo and Mottron, Laurent and Oh, Miae},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

The code is licensed under the MIT License. See [LICENSE](LICENSE) for the full text. Please cite the paper when using this code.

## Contact

Correspondence to Miae Oh (miae612@khu.ac.kr).
