#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${REPORT_LLM_DATA_ROOT:-$HOME/Dropbox/ReportLLM_analysis}"
python "$SCRIPT_DIR/roberta_sentence_attention_run_ko.py" \
  --experiment_name asd_classifier \
  --tokenized_path "${TOKENIZED_PATH:-$DATA_ROOT/pipeline/1_classifier/data/489reports/reports_tokenized_153stc_128tkn_no_headings}" \
  --epochs 40 --patience 10 --batch_size 1 --lr 1e-5 \
  --inner_val_splits 10 --seed 42 --save_fold_models --save_intermediates \
  --report_max_length 153 --sentence_max_length 128 --no_wandb
