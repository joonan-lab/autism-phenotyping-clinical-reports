"""External private inputs and analysis outputs."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("REPORT_LLM_DATA_ROOT", "~/Dropbox/ReportLLM_analysis")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("REPORT_LLM_OUTPUT_ROOT", str(DATA_ROOT / "02_results"))).expanduser()
INPUT_ROOT = DATA_ROOT / "00_inputs"
INTER = Path(os.environ.get("CLASSIFIER_INTERMEDIATES_DIR", str(DATA_ROOT / "pipeline/1_classifier/intermediates/asd_classifier"))).expanduser()
HEADS = DATA_ROOT / "pipeline/1_classifier/heads"
SILVER = Path(os.environ.get("SILVER_LABELS_JSONL", str(INPUT_ROOT / "step2_silver_labeling/silver_labels_489reports.jsonl"))).expanduser()
GOLD = Path(os.environ.get("GOLD_LABELS_JSONL", str(INPUT_ROOT / "step2_silver_labeling/gold_labels/gold_labels_26reports.jsonl"))).expanduser()
VECTORS = Path(os.environ.get("DOMAIN_VECTOR_OUTPUT_DIR", str(OUTPUT_ROOT / "domain_vectors"))).expanduser()
SPACE = OUTPUT_ROOT / "phenotype_space"
EVIDENCE = OUTPUT_ROOT / "domain_contributions"
SILVER_RESULTS = OUTPUT_ROOT / "validation/silver_labels"
CLASSIFIER_RESULTS = OUTPUT_ROOT / "validation/classifier"
E_RESULTS = OUTPUT_ROOT / "validation/domain_contributions"
LENGTH_RESULTS = OUTPUT_ROOT / "validation/report_length"
FIGURE_SOURCE = Path(os.environ.get("REPORT_LLM_FIGURE_SOURCE_DIR", str(DATA_ROOT / "figure_source"))).expanduser()
TABLE_SOURCE = Path(os.environ.get("REPORT_LLM_TABLE_SOURCE_DIR", str(DATA_ROOT / "table_source"))).expanduser()
ONTOLOGY = ROOT / "ontology/domains_19.yaml"
def out(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
