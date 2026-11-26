#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${REPORT_LLM_DATA_ROOT:-$HOME/Dropbox/ReportLLM_analysis}"

python "$SCRIPT_DIR/preprocess_data_ko.py" \
  --input_type txt \
  --txt_dir "${TXT_DIR:-$DATA_ROOT/00_inputs/step1_nlp_asd/489reports/reports_txt_no_headings}" \
  --meta_path "${META_PATH:-$DATA_ROOT/00_inputs/step1_nlp_asd/metadata/metadata_489reports.csv}" \
  --model_name "${MODEL_NAME:-klue/roberta-base}" \
  --output_dir "${OUTPUT_DIR:-$DATA_ROOT/pipeline/1_classifier/data/489reports/reports_tokenized_153stc_128tkn_no_headings}" \
  --report_max_length "${REPORT_MAX_LENGTH:-153}" \
  --sentence_max_length "${SENTENCE_MAX_LENGTH:-128}"
