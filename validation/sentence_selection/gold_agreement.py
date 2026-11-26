"""Calculate Table S3 from private sentence labels and stored OOF attention."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(REPO))
import analysis_paths as paths


def read_labels(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_mass(labels, domain_index):
    """Allocate one sentence's mass equally across its assigned domains."""
    domains = [item["domain_id"] if isinstance(item, dict) else item for item in labels]
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("Sentence labels must contain distinct, nonempty domain assignments")
    mass = np.zeros(len(domain_index), dtype=float)
    for domain in domains:
        mass[domain_index[domain]] = 1.0 / len(domains)
    return mass


def build_profiles(intermediates, silver_path, gold_path, domain_meta):
    """Match gold by exact text, then build full and attention-selected profiles."""
    inter = Path(intermediates)
    ids = np.load(inter / "report_id_array.npy", allow_pickle=True).astype(str)
    people = np.load(inter / "patient_id_array.npy", allow_pickle=True).astype(str)
    mask = np.load(inter / "valid_sentence_mask_np.npy").astype(bool)
    attention = np.load(inter / "attention_matrices_np.npy", mmap_mode="r")
    if ids.ndim != 1 or len(set(ids)) != len(ids) or people.shape != ids.shape:
        raise ValueError("Report and individual IDs must be unique/aligned report arrays")
    if mask.ndim != 2 or mask.shape[0] != len(ids) or attention.shape != (*mask.shape, mask.shape[1]):
        raise ValueError("Attention and eligible-sentence masks have incompatible shapes")
    with Path(domain_meta).open(encoding="utf-8") as handle:
        domains = json.load(handle)["domain_columns"]
    if len(domains) != 19 or len(set(domains)) != len(domains):
        raise ValueError("Table S3 requires the same ordered nineteen-domain ontology")
    domain_index = {name: index for index, name in enumerate(domains)}
    gold = read_labels(gold_path)
    gold_ids = sorted({str(row["report_id"]) for row in gold})
    report_index = {rid: index for index, rid in enumerate(ids)}
    if not gold_ids or not set(gold_ids).issubset(report_index):
        raise ValueError("Every gold report must have stored classifier inputs")
    silver_by_text, gold_by_text = defaultdict(list), defaultdict(list)
    for row in read_labels(silver_path):
        if str(row["report_id"]) in set(gold_ids):
            silver_by_text[(str(row["report_id"]), row["sentence"].strip())].append(row)
    for row in gold:
        gold_by_text[(str(row["report_id"]), row["sentence"].strip())].append(row)
    if set(silver_by_text) != set(gold_by_text):
        raise ValueError("Gold and silver must cover the same sentence texts within each gold report")
    n_sentences = mask.shape[1]
    silver_mass = {rid: np.zeros((n_sentences, len(domains))) for rid in gold_ids}
    gold_mass = {rid: np.zeros((n_sentences, len(domains))) for rid in gold_ids}
    seen = set()
    for key, gold_rows in gold_by_text.items():
        silver_rows = silver_by_text[key]
        if len(gold_rows) != len(silver_rows):
            raise ValueError("Repeated sentence texts have unequal gold/silver counts")
        if len(gold_rows) > 1:
            assignments = {tuple(sorted(row["labels"])) for row in gold_rows}
            if len(assignments) != 1:
                raise ValueError("Repeated sentence text has conflicting gold assignments")
        for gold_row, silver_row in zip(
            sorted(gold_rows, key=lambda row: row["sentence_idx"]),
            sorted(silver_rows, key=lambda row: row["sentence_idx"]),
        ):
            rid = key[0]
            j = int(silver_row["sentence_idx"])
            i = report_index[rid]
            if j < 0 or j >= n_sentences or not mask[i, j] or (rid, j) in seen:
                raise ValueError("Silver sentence indices must map once to eligible classifier positions")
            seen.add((rid, j))
            silver_mass[rid][j] = label_mass(silver_row["labels"], domain_index)
            gold_mass[rid][j] = label_mass(gold_row["labels"], domain_index)
    profiles = {"full": [], "attention_top10": []}
    gold_profiles = []
    for rid in gold_ids:
        i = report_index[rid]
        valid = np.flatnonzero(mask[i])
        if not len(valid):
            raise ValueError("Reports must have at least one eligible sentence")
        for mass in (silver_mass[rid], gold_mass[rid]):
            if not np.allclose(mass.sum(-1), mask[i]):
                raise ValueError("Sentence annotations must cover every eligible position in a gold report")
        # Same valid-query/key attention and stable source-index tie order as v4.1.
        weights = np.asarray(attention[i])[np.ix_(valid, valid)].mean(0)
        if not np.isfinite(weights).all():
            raise ValueError("Eligible attention weights must be finite")
        selected = valid[np.argsort(-weights, kind="stable")[: min(10, len(valid))]]
        profiles["full"].append(silver_mass[rid].sum(0) / len(valid))
        profiles["attention_top10"].append(silver_mass[rid][selected].sum(0) / len(selected))
        gold_profiles.append(gold_mass[rid].sum(0) / len(valid))
    return (
        {method: np.asarray(values) for method, values in profiles.items()},
        np.asarray(gold_profiles),
        people[[report_index[rid] for rid in gold_ids]],
    )


def cosine(x, y):
    return np.sum(x * y, axis=-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1))


def spearman(x, y):
    x, y = rankdata(x, axis=-1), rankdata(y, axis=-1)
    x, y = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return np.divide((x * y).sum(-1), denominator, out=np.full(x.shape[:-1], np.nan), where=denominator > 1e-15)


def mean_summary(values, people):
    """Calculate percentile intervals from 1,000 individual-level bootstrap resamples using seed 42."""
    valid = np.isfinite(values)
    per, people = np.asarray(values)[valid], np.asarray(people)[valid]
    if not len(per):
        raise ValueError("Metric has no finite report values")
    groups = np.unique(people)
    lookup = {person: np.flatnonzero(people == person) for person in groups}
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(1000):
        indices = np.concatenate([lookup[person] for person in rng.choice(groups, len(groups), replace=True)])
        boot.append(per[indices].mean())
    return dict(
        mean=float(per.mean()), median=float(np.median(per)), n_reports=len(per),
        n_individuals=len(groups), n_valid=len(boot),
        percentile_interval=json.dumps(np.percentile(boot, [2.5, 97.5]).tolist()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediates", type=Path, default=paths.INTER)
    parser.add_argument("--silver", type=Path, default=paths.SILVER)
    parser.add_argument("--gold", type=Path, default=paths.GOLD)
    parser.add_argument("--domain-meta", type=Path, default=paths.VECTORS / "domain_vectors_meta_latest.json")
    parser.add_argument("--output", type=Path, default=paths.OUTPUT_ROOT / "validation/sentence_selection/gold_sentence_selection_summary.tsv")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(REPO.resolve()):
        raise ValueError("Write generated summaries outside the source checkout")
    profiles, gold, people = build_profiles(args.intermediates, args.silver, args.gold, args.domain_meta)
    rows = []
    for method, values in profiles.items():
        metrics = {
            "cosine_full_gold": cosine(values, gold),
            "TV_full_gold": 0.5 * np.abs(values - gold).sum(-1),
            "rank_correlation_full_gold": spearman(values, gold),
        }
        for metric, value in metrics.items():
            rows.append(dict(metric=metric, method=method, **mean_summary(value, people)))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    print(f"Wrote Table S3 summary for {len(gold)} gold reports to {output}")


if __name__ == "__main__":
    main()
