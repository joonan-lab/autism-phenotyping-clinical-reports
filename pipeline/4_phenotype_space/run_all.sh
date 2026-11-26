#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The selected example is step 12 and follows domain-contribution calculation.
for script in confound_check/report_type_figure_data.py step1_effective_dim.py step2_pseudocount_sensitivity.py step3_gmm_vs_heavytail.py step4_residual_tail.py step5_seed_stability.py step6_bootstrap_permutation.py step7_normative.py step8_domain_deviation_prevalence.py step9_variance.py step10_atype_only_mode_stability.py step11_ados_variance_share.py; do
  python "$DIR/$script"
done
