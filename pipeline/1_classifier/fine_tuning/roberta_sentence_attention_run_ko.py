"""
한국어 임상 보고서용 Sentence Attention BERT 학습 스크립트 (v4.1)

v4.1 변경사항:
- AUROC 동점은 개선 없음으로 처리하고 실제 checkpoint 갱신 시에만 best epoch 기록
- 매 epoch 보고서 순서를 무작위로 섞되 입력/마스크/라벨 정렬 유지

v4 변경사항:
- 각 outer fold의 train에서 inner validation(개인 단위, 1/inner_val_splits)을 분리해 early stopping에 사용
- outer held-out fold는 best epoch 모델로 학습 종료 후 1회만 평가 (성능 지표, OOS 산출물 모두 이 결과)

v3 변경사항:
- 검증 분석용 OOS 산출물 저장 확장: probs, report/patient ID, valid sentence mask/count, fold split/checkpoint manifest
- 후속 attention validity 및 domain-vector robustness 분석의 row alignment와 padding masking 재현성 보강

v2.5 변경사항:
1. Padding 문장에 대한 attention masking 적용
- sentence_attention_base_pool_ko.py 모델 사용
- padding 문장에 attention이 부여되지 않도록 수정

v2.4 기능 유지:
- 폴드별 모델 저장 옵션 (--save_fold_models)
- Out-of-sample inference 결과 및 중간 산물 저장 기능
- Early Stopping (validation AUROC 기준)
- 다양한 메트릭 (AUROC, AUPRC, Precision, Recall, Specificity, F1, Accuracy)

실행 예시:
    python roberta_sentence_attention_run_ko_v4_1.py --epochs 10 --patience 3 --inner_val_splits 10 --save_fold_models
"""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths

### Imports ###

import os
import sys
import random
import numpy as np
import pandas as pd
import copy
from datetime import datetime
import json

import torch
from torch import nn


def set_seed(seed=42):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from transformers import AutoTokenizer, AutoConfig
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix
)

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.insert(0, str(_REPO / "pipeline/1_classifier"))

from helpers.batcher import create_batches
from helpers.hierarchical_tokenizer_ko import hierarchical_tokenizer_korean
import glob
# v2.5: padding masking이 적용된 새 모델 사용 (v1.3.1)
from custom_models.sentence_attention_base_pool_ko import SentenceAttentionBERTKorean

try:
    import wandb
except ImportError:
    wandb = None
import argparse

# ============================================================================
# 기본 설정
# ============================================================================
BASE_MODEL_NAME = "klue/roberta-base"
ROOT = str(paths.DATA_ROOT / "pipeline/1_classifier")
DATA_PATH = str(paths.INPUT_ROOT / "step1_nlp_asd")
TYPE_PATH = os.path.join(DATA_PATH, "all1st309")
REPORT_PATH = os.environ.get("REPORT_TXT_DIR", str(paths.INPUT_ROOT / "step1_nlp_asd/489reports/reports_txt_no_headings"))


def calculate_all_metrics(probs, labels):
    """모든 평가 메트릭 계산"""
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)

    metrics = {}

    # AUROC
    try:
        metrics['auroc'] = roc_auc_score(labels, probs)
    except:
        metrics['auroc'] = 0.5

    # AUPRC
    try:
        metrics['auprc'] = average_precision_score(labels, probs)
    except:
        metrics['auprc'] = 0.5

    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, preds)

    # Precision, Recall, F1
    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['recall'] = recall_score(labels, preds, zero_division=0)
    metrics['f1'] = f1_score(labels, preds, zero_division=0)

    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Confusion Matrix 값 추가
    metrics['TP'] = int(tp)
    metrics['FP'] = int(fp)
    metrics['TN'] = int(tn)
    metrics['FN'] = int(fn)

    return metrics


class EarlyStopping:
    """Early Stopping 클래스 (AUROC 기준, attention 결과 저장 포함)"""

    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
            patience: 개선이 없을 때 기다리는 epoch 수
            min_delta: 개선으로 인정할 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_attention_results = None  # attention 결과 저장용

    def __call__(self, auroc, model, attention_results=None):
        """AUROC 기준 (높을수록 좋음). Checkpoint 갱신 여부를 반환."""
        if self.best_score is None:
            self.best_score = auroc
            self.save_checkpoint(model, attention_results)
            return True
        elif auroc <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = auroc
            self.save_checkpoint(model, attention_results)
            self.counter = 0
            return True

    def save_checkpoint(self, model, attention_results=None):
        """Best model state 및 attention 결과 저장"""
        if isinstance(model, nn.DataParallel):
            self.best_model_state = copy.deepcopy(model.module.state_dict())
        else:
            self.best_model_state = copy.deepcopy(model.state_dict())

        # attention 결과도 저장 (deep copy)
        if attention_results is not None:
            self.best_attention_results = {
                'logits': attention_results['logits'].copy(),
                'attn_weights': attention_results['attn_weights'].copy(),
                'attn_outputs': attention_results['attn_outputs'].copy(),
                'lhs_embs': attention_results['lhs_embs'].copy(),
                'layer_pooled_embs': attention_results['layer_pooled_embs'].copy(),
            }


def load_data(data_mode, model_name, sentence_max_length, report_max_length,
              csv_path=None, meta_csv_path=None, tokenized_path=None):
    """데이터 로드 함수"""

    if data_mode == "tensor":
        if tokenized_path is None:
            tokenized_path = os.path.join(DATA_PATH, "reports_tokenized")  # 기본값
        input_tensor = torch.load(os.path.join(tokenized_path, "input_tensor"), weights_only=False)
        attention_mask_tensor = torch.load(os.path.join(tokenized_path, "attention_mask_tensor"), weights_only=False)
        label_tensor = torch.load(os.path.join(tokenized_path, "label_tensor"), weights_only=False)
        report_id_array = torch.load(os.path.join(tokenized_path, "report_id_array"), weights_only=False)
    elif data_mode == "txt":
        if meta_csv_path is None:
            raise ValueError("data_mode='txt'일 때 --meta_csv_path를 지정해야 합니다.")
        report_files = glob.glob(os.path.join(REPORT_PATH, "*.txt"))
        meta_frame = pd.read_csv(meta_csv_path, encoding="utf-8")
        input_tensor, attention_mask_tensor, label_tensor, report_id_array = hierarchical_tokenizer_korean(
            report_files=report_files, meta_frame=meta_frame, tokenizer_name=model_name,
            sentence_max_length=sentence_max_length, sentence_min_length=30,
            report_max_length=report_max_length
        )
    else:
        raise ValueError(f"지원하지 않는 data_mode: {data_mode}")

    return input_tensor, attention_mask_tensor, label_tensor, report_id_array


def evaluate(model, data, mask, labels, loss_fn, collect_attention=False):
    """배치 단위 평가. collect_attention=True이면 OOS 저장용 attention 결과도 반환"""
    model.eval()
    losses, probs_all, labels_all = [], [], []
    logits_all, attn_weights_all, attn_outputs_all = [], [], []
    lhs_embs_all, layer_pooled_embs_all = [], []

    with torch.no_grad():
        for batch_idx in range(len(data)):
            current_batch = data[batch_idx]
            current_mask = mask[batch_idx]

            # DataParallel은 리스트 출력(last_hidden_embs, all_layer_embs)을 제대로 처리하지 못함
            if isinstance(model, nn.DataParallel):
                output, attn_weights, attn_output, last_hidden_embs, all_layer_embs = model.module(current_batch, attn_mask=current_mask)
            else:
                output, attn_weights, attn_output, last_hidden_embs, all_layer_embs = model(current_batch, attn_mask=current_mask)

            output = torch.atleast_1d(output)
            labels_true = torch.atleast_1d(labels[batch_idx])
            losses.append(loss_fn(output, labels_true).item())
            probs_all.extend(torch.sigmoid(output).cpu().numpy())
            labels_all.extend(labels_true.cpu().numpy())

            if collect_attention:
                logits_all.append(output.cpu().numpy())
                attn_weights_all.append(attn_weights.cpu().numpy())
                attn_outputs_all.append(attn_output.cpu().numpy())
                for lhs in last_hidden_embs:
                    lhs_embs_all.append(lhs.cpu().numpy())
                for lp in all_layer_embs:
                    layer_pooled_embs_all.append(lp.cpu().numpy())

    metrics = calculate_all_metrics(probs_all, labels_all)
    metrics['loss'] = np.mean(losses)

    attention_results = None
    if collect_attention:
        attention_results = {
            'logits': np.concatenate(logits_all),
            'attn_weights': np.concatenate(attn_weights_all),
            'attn_outputs': np.concatenate(attn_outputs_all),
            'lhs_embs': np.stack(lhs_embs_all, axis=0),
            'layer_pooled_embs': np.stack(layer_pooled_embs_all, axis=0),
        }
    return metrics, attention_results


def train_single_fold(fold_idx, train_idx, inner_val_idx, test_idx, input_tensor, attention_mask_tensor,
                      label_tensor, config, device, use_wandb=False):
    """단일 Fold 학습: inner validation AUROC로 early stopping, outer test fold는 종료 후 1회 평가"""

    batch_size = config['batch_size']

    # 데이터 배치 생성
    valid_data = create_batches(input_tensor[inner_val_idx], batch_size)
    valid_mask = create_batches(attention_mask_tensor[inner_val_idx], batch_size)
    valid_labels = create_batches(label_tensor[inner_val_idx], batch_size)
    test_data = create_batches(input_tensor[test_idx], batch_size)
    test_mask = create_batches(attention_mask_tensor[test_idx], batch_size)
    test_labels = create_batches(label_tensor[test_idx], batch_size)

    # 모델 초기화 (v2.5.1: padding masking이 적용된 모델)
    model = SentenceAttentionBERTKorean(
        base_model_name=config['model_name'],
        report_max_length=config['report_max_length'],
        ff_dropout=config['ff_dropout'],
        att_dropout=config['att_dropout'],
        class_dropout=config['class_dropout'],
        sentence_dropout=config['sentence_dropout']
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # 옵티마이저 설정
    if config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    loss_fn = nn.BCEWithLogitsLoss()

    # Early Stopping 초기화 (inner validation AUROC 기준)
    early_stopping = EarlyStopping(patience=config['patience'])
    best_epoch = 0

    # 에폭 학습
    for epoch in range(config['epochs']):
        model.train()
        # 보고서 단위 shuffle: 동일 인덱스로 입력, 마스크, 라벨을 함께 재배치
        epoch_train_idx = np.asarray(train_idx)[np.random.permutation(len(train_idx))]
        train_data = create_batches(input_tensor[epoch_train_idx], batch_size)
        train_mask = create_batches(attention_mask_tensor[epoch_train_idx], batch_size)
        train_labels = create_batches(label_tensor[epoch_train_idx], batch_size)
        train_losses = []
        train_probs_all = []
        train_labels_all = []

        for train_batch_idx in range(len(train_data)):
            model.zero_grad()
            current_batch = train_data[train_batch_idx]
            current_mask = train_mask[train_batch_idx]

            # GPU당 배치가 1개 이하면 DataParallel 우회 (스칼라 출력 문제 방지)
            if isinstance(model, nn.DataParallel) and current_batch.size(0) < torch.cuda.device_count() * 2:
                output, _, _, _, _ = model.module(current_batch, attn_mask=current_mask)
            else:
                output, _, _, _, _ = model(current_batch, attn_mask=current_mask)

            output = torch.atleast_1d(output)
            train_labels_true = torch.atleast_1d(train_labels[train_batch_idx])
            loss = loss_fn(output, train_labels_true)

            train_losses.append(loss.item())
            train_probs_all.extend(torch.sigmoid(output).cpu().detach().numpy())
            train_labels_all.extend(train_labels_true.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

        # 에폭 끝 inner validation
        train_metrics = calculate_all_metrics(train_probs_all, train_labels_all)
        train_metrics['loss'] = np.mean(train_losses)
        valid_metrics, _ = evaluate(model, valid_data, valid_mask, valid_labels, loss_fn)

        print(f"  Epoch {epoch+1}: Train AUROC={train_metrics['auroc']:.4f}, "
              f"Inner-valid AUROC={valid_metrics['auroc']:.4f}, Inner-valid Acc={valid_metrics['accuracy']:.4f}")

        if use_wandb:
            wandb.log({
                "train_loss": train_metrics['loss'],
                "train_auroc": train_metrics['auroc'],
                "valid_loss": valid_metrics['loss'],
                "valid_auroc": valid_metrics['auroc'],
                "valid_accuracy": valid_metrics['accuracy'],
            })

        checkpoint_updated = early_stopping(valid_metrics['auroc'], model)

        if checkpoint_updated:
            best_epoch = epoch + 1

        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Best model state 복원
    if early_stopping.best_model_state is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(early_stopping.best_model_state)
        else:
            model.load_state_dict(early_stopping.best_model_state)

    # Outer test fold 1회 평가
    test_metrics, attention_results = evaluate(model, test_data, test_mask, test_labels, loss_fn,
                                               collect_attention=True)
    print(f"  Best inner-valid epoch={best_epoch} (AUROC={early_stopping.best_score:.4f}) -> "
          f"Outer test AUROC={test_metrics['auroc']:.4f}, Acc={test_metrics['accuracy']:.4f}")

    if use_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["inner_valid_auroc_best"] = early_stopping.best_score
        wandb.summary["outer_test_auroc"] = test_metrics['auroc']

    return {
        'fold': fold_idx + 1,
        'best_epoch': best_epoch,
        'inner_valid_auroc': early_stopping.best_score,
        'metrics': test_metrics,
        'model_state': early_stopping.best_model_state,
        'attention_results': attention_results,
    }


def get_patient_based_split(pat_ids, label_tensor, test_ratio, seed):
    """환자 기반 Hold-out 분할"""
    from sklearn.model_selection import train_test_split

    unique_pats = np.unique(pat_ids)

    # 환자별 레이블 (다수결)
    pat_labels = []
    for pat in unique_pats:
        pat_mask = np.array(pat_ids) == pat
        pat_label = int(label_tensor[pat_mask].float().mean() >= 0.5)
        pat_labels.append(pat_label)
    pat_labels = np.array(pat_labels)

    # 환자 단위로 split
    pats_trainval, pats_test = train_test_split(
        unique_pats, test_size=test_ratio, stratify=pat_labels, random_state=seed
    )

    # 인덱스로 변환
    trainval_idx = np.where(np.isin(pat_ids, pats_trainval))[0]
    test_idx = np.where(np.isin(pat_ids, pats_test))[0]

    return trainval_idx, test_idx


def train_model(config, use_wandb=False, verbose=True):
    if use_wandb and wandb is None:
        raise ImportError("Install wandb or run with --no_wandb")
    """전체 K-Fold 학습 실행 함수"""

    # 시드 고정 (재현성)
    seed = config.get('seed', 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 로드
    input_tensor, attention_mask_tensor, label_tensor, report_id_array = load_data(
        data_mode=config.get('data_mode', 'tensor'),
        model_name=config['model_name'],
        sentence_max_length=config.get('sentence_max_length', 64),
        report_max_length=config.get('report_max_length', 64),
        meta_csv_path=config.get('meta_csv_path'),
        tokenized_path=config.get('tokenized_path')
    )

    # 환자 ID 추출
    pat_ids = np.array([str(rid)[:12] for rid in report_id_array])

    if verbose:
        print(f"총 보고서 수: {len(report_id_array)}, 환자 수: {len(np.unique(pat_ids))}")

    # Hold-out test set 분리 (test_ratio > 0인 경우)
    test_ratio = config.get('test_ratio', 0.0)
    test_idx = None
    test_metrics = None

    report_id_array = np.asarray(report_id_array)

    if test_ratio > 0:
        trainval_idx, test_idx = get_patient_based_split(pat_ids, label_tensor, test_ratio, seed)
        if verbose:
            print(f"Hold-out 분할: Train+Valid={len(trainval_idx)}, Test={len(test_idx)}")
        # Train+Valid만 사용
        input_tensor_cv = input_tensor[trainval_idx]
        attention_mask_tensor_cv = attention_mask_tensor[trainval_idx]
        label_tensor_cv = label_tensor[trainval_idx]
        pat_ids_cv = pat_ids[trainval_idx]
        report_id_array_cv = report_id_array[trainval_idx]
        cv_original_indices = trainval_idx  # 원본 인덱스 저장
    else:
        input_tensor_cv = input_tensor
        attention_mask_tensor_cv = attention_mask_tensor
        label_tensor_cv = label_tensor
        pat_ids_cv = pat_ids
        report_id_array_cv = report_id_array
        cv_original_indices = np.arange(len(report_id_array))  # 원본 인덱스 저장

    valid_sentence_mask_np = attention_mask_tensor_cv.cpu().numpy().sum(axis=-1) > 2
    n_valid_sentences_np = valid_sentence_mask_np.sum(axis=1).astype(np.int64)

    # Cross-validation 설정
    skf = StratifiedGroupKFold(n_splits=config.get('k_fold_CV', 5), shuffle=True, random_state=seed)
    splits = list(skf.split(input_tensor_cv, label_tensor_cv, groups=pat_ids_cv))

    # Inner split: outer train에서 개인 단위로 1/inner_val_splits를 early stopping용으로 분리
    inner_splits = []
    for train_idx, _ in splits:
        inner_skf = StratifiedGroupKFold(n_splits=config['inner_val_splits'], shuffle=True, random_state=seed)
        inner_tr_rel, inner_val_rel = next(inner_skf.split(
            train_idx, np.asarray(label_tensor_cv)[train_idx], groups=pat_ids_cv[train_idx]))
        inner_splits.append((train_idx[inner_tr_rel], train_idx[inner_val_rel]))

    fold_split_manifest = []
    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        inner_train_idx, inner_val_idx = inner_splits[fold_idx]
        fold_split_manifest.append({
            "fold": fold_idx + 1,
            "train_indices_cv": train_idx.tolist(),
            "valid_indices_cv": valid_idx.tolist(),
            "train_original_indices": cv_original_indices[train_idx].tolist(),
            "valid_original_indices": cv_original_indices[valid_idx].tolist(),
            "train_report_ids": report_id_array_cv[train_idx].astype(str).tolist(),
            "valid_report_ids": report_id_array_cv[valid_idx].astype(str).tolist(),
            "train_patient_ids": pat_ids_cv[train_idx].astype(str).tolist(),
            "valid_patient_ids": pat_ids_cv[valid_idx].astype(str).tolist(),
            "inner_train_indices_cv": inner_train_idx.tolist(),
            "inner_valid_indices_cv": inner_val_idx.tolist(),
            "inner_valid_report_ids": report_id_array_cv[inner_val_idx].astype(str).tolist(),
        })

    # GPU로 이동
    input_tensor_cv = input_tensor_cv.to(device)
    attention_mask_tensor_cv = attention_mask_tensor_cv.to(device)
    label_tensor_cv = label_tensor_cv.float().to(device)

    if test_ratio > 0:
        input_tensor_test = input_tensor[test_idx].to(device)
        attention_mask_test = attention_mask_tensor[test_idx].to(device)
        label_tensor_test = label_tensor[test_idx].float().to(device)

    all_fold_results = []
    best_fold_model_state = None
    best_fold_auroc = 0.0

    # Out-of-sample inference 결과 저장용 배열 초기화
    n_samples = len(input_tensor_cv)
    #report_max_length = config.get('report_max_length', 64)
    num_sentences = input_tensor_cv.shape[1]  # 실제 문장 수
    hidden_size = 768  # KLUE RoBERTa base의 hidden size

    oos_logits = np.zeros(n_samples)
    oos_probs = np.zeros(n_samples)
    oos_attention_matrices = np.zeros((n_samples, num_sentences, num_sentences))
    oos_attention_weighted_embs = np.zeros((n_samples, num_sentences, hidden_size))

    num_layers = 13  # embedding + 12 transformer layers
    sentence_max_length = config.get('sentence_max_length', 64)
    oos_lhs_embs = np.zeros((n_samples, num_sentences, sentence_max_length, hidden_size))
    oos_layer_pooled_embs = np.zeros((n_samples, num_layers, hidden_size))

    oos_labels = label_tensor_cv.cpu().numpy()
    oos_fold_assignment = np.zeros(n_samples, dtype=int)

    # 폴드별 모델 저장 경로
    model_dir = os.path.join(ROOT, "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    fold_checkpoint_manifest = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{config.get('k_fold_CV', 5)}")
            print(f"{'='*50}")

        # wandb 초기화
        if use_wandb:
            wandb.init(
                project=config.get('experiment_name') or config.get('wandb_project', 'NLP-ASD-Korean'),
                name=f"GridSearch_Fold{fold_idx + 1}",
                group=f"lr={config['lr']}_dropout={config['ff_dropout']}",
                config=config,
                reinit=True
            )

        inner_train_idx, inner_val_idx = inner_splits[fold_idx]
        if verbose:
            print(f"Inner train={len(inner_train_idx)}, Inner valid={len(inner_val_idx)}, Outer test={len(valid_idx)}")

        result = train_single_fold(
            fold_idx, inner_train_idx, inner_val_idx, valid_idx,
            input_tensor_cv, attention_mask_tensor_cv, label_tensor_cv,
            config, device, use_wandb
        )
        all_fold_results.append(result)

        # Best fold 모델 저장 (test 평가용)
        if result['metrics']['auroc'] > best_fold_auroc:
            best_fold_auroc = result['metrics']['auroc']
            best_fold_model_state = result['model_state']

        fold_model_file = None

        # 폴드별 모델 파일 저장
        if config.get('save_fold_models', False) and result['model_state'] is not None:
            exp_name = config.get('experiment_name', 'model')
            fold_model_file = os.path.join(model_dir, f"{exp_name}_fold{fold_idx + 1}.pt")
            torch.save(result['model_state'], fold_model_file)
            if verbose:
                print(f"  폴드 모델 저장: {fold_model_file}")

        fold_checkpoint_manifest.append({
            "fold": fold_idx + 1,
            "checkpoint_path": fold_model_file,
            "saved": fold_model_file is not None,
        })

        # Out-of-sample attention 결과 저장 (train_single_fold에서 이미 계산됨)
        if config.get('save_intermediates', False) or config.get('save_fold_models', False):
            attn_results = result.get('attention_results')
            if attn_results is not None:
                # 전체 결과 배열에 저장 (valid_idx 위치에)
                for i, idx in enumerate(valid_idx):
                    oos_logits[idx] = attn_results['logits'][i]
                    oos_probs[idx] = 1 / (1 + np.exp(-attn_results['logits'][i]))  # sigmoid
                    oos_attention_matrices[idx] = attn_results['attn_weights'][i]
                    oos_attention_weighted_embs[idx] = attn_results['attn_outputs'][i]
                    oos_lhs_embs[idx] = attn_results['lhs_embs'][i]
                    oos_layer_pooled_embs[idx] = attn_results['layer_pooled_embs'][i]
                    oos_fold_assignment[idx] = fold_idx + 1

                if verbose:
                    print(f"  Best epoch attention 결과 저장 완료: {len(valid_idx)}개 샘플")

        # wandb 종료
        if use_wandb:
            wandb.finish()

    # 결과 요약
    metric_names = ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'loss']
    summary = {}
    for m in metric_names:
        values = [r['metrics'][m] for r in all_fold_results]
        summary[f'avg_{m}'] = np.mean(values)
        summary[f'std_{m}'] = np.std(values)

    # Hold-out test set 평가 (test_ratio > 0인 경우)
    if test_ratio > 0 and best_fold_model_state is not None:
        if verbose:
            print(f"\n{'='*50}")
            print("Hold-out Test Set 평가")
            print(f"{'='*50}")

        # Best fold 모델 로드
        model = SentenceAttentionBERTKorean(
            base_model_name=config['model_name'],
            report_max_length=config['report_max_length'],
            ff_dropout=config['ff_dropout'],
            att_dropout=config['att_dropout'],
            class_dropout=config['class_dropout'],
            sentence_dropout=config['sentence_dropout']
        )
        model.load_state_dict(best_fold_model_state)
        model = model.to(device)
        model.eval()

        # Test 평가
        test_data = create_batches(input_tensor_test, config['batch_size'])
        test_mask = create_batches(attention_mask_test, config['batch_size'])
        test_labels = create_batches(label_tensor_test, config['batch_size'])

        test_probs_all = []
        test_labels_all = []

        with torch.no_grad():
            for batch_idx in range(len(test_data)):
                current_batch = test_data[batch_idx]
                current_mask = test_mask[batch_idx]

                # GPU당 배치가 1개 이하면 DataParallel 우회
                if isinstance(model, nn.DataParallel) and current_batch.size(0) < torch.cuda.device_count() * 2:
                    output, _, _, _, _ = model.module(current_batch, attn_mask=current_mask)
                else:
                    output, _, _, _, _ = model(current_batch, attn_mask=current_mask)

                output = torch.atleast_1d(output)
                test_probs_all.extend(torch.sigmoid(output).cpu().numpy())
                test_labels_all.extend(torch.atleast_1d(test_labels[batch_idx]).cpu().numpy())

        test_metrics = calculate_all_metrics(test_probs_all, test_labels_all)

        if verbose:
            print(f"Test AUROC: {test_metrics['auroc']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")

    # Out-of-sample inference 결과 포함
    oos_results = None
    if config.get('save_intermediates', False) or config.get('save_fold_models', False):
        predicted_labels_np = (oos_probs >= 0.5).astype(np.int64)

        oos_results = {
            'logits_np': oos_logits,
            'probs_np': oos_probs,
            'predicted_labels_np': predicted_labels_np,
            'attention_matrices_np': oos_attention_matrices,
            'attention_weighted_sentence_embs_np': oos_attention_weighted_embs,
            'lhs_embs_np': oos_lhs_embs,                    # 추가
            'layer_pooled_embs_np': oos_layer_pooled_embs,  # 추가
            'labels_np': oos_labels,
            'fold_assignment': oos_fold_assignment,
            "report_id_array": report_id_array_cv.astype(str),
            "patient_id_array": pat_ids_cv.astype(str),
            "cv_original_indices": cv_original_indices,
            "valid_sentence_mask_np": valid_sentence_mask_np,
            "n_valid_sentences_np": n_valid_sentences_np,
        }

    return {
        'fold_results': all_fold_results,
        'summary': summary,
        'config': config,
        'test_metrics': test_metrics,
        'best_model_state': best_fold_model_state,
        'splits': splits,
        'label_tensor_cv': label_tensor_cv.cpu(),
        'oos_results': oos_results,
        "fold_split_manifest": fold_split_manifest,
        "fold_checkpoint_manifest": fold_checkpoint_manifest,
    }


def main():
    ### Argument Parser ###
    parser = argparse.ArgumentParser(description='Korean Clinical Report Classification Training')

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", default=10, type=int, help="최대 학습 에폭 수")
    parser.add_argument("--batch_size", default=8, type=int, help="배치 크기")
    parser.add_argument("--lr", default=1e-5, type=float, help="학습률")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="옵티마이저")
    parser.add_argument("--patience", default=3, type=int, help="Early stopping patience")

    # Dropout 설정
    parser.add_argument("--ff_dropout", default=0.1, type=float)
    parser.add_argument("--att_dropout", default=0.1, type=float)
    parser.add_argument("--class_dropout", default=0.1, type=float)
    parser.add_argument("--sentence_dropout", default=0.0, type=float)

    # 기타 설정
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--job_id", default="0", type=str)
    parser.add_argument("--k_fold_CV", default=5, type=int)
    parser.add_argument("--inner_val_splits", default=10, type=int,
                        help="outer train 중 1/N을 inner validation(early stopping)으로 사용")
    parser.add_argument("--model_name", default=BASE_MODEL_NAME, type=str)
    parser.add_argument("--report_max_length", default=64, type=int)
    parser.add_argument("--sentence_max_length", default=64, type=int)

    # 데이터 설정
    parser.add_argument("--data_mode", default="tensor", type=str)
    parser.add_argument("--meta_csv_path", default=None, type=str)
    parser.add_argument("--tokenized_path", default=None, type=str,
                        help="토큰화된 텐서 데이터 경로 (data_mode='tensor'일 때 사용)")

    # 재현성 및 실험 설정
    parser.add_argument("--seed", default=42, type=int, help="랜덤 시드")
    parser.add_argument("--test_ratio", default=0.0, type=float,
                        help="Hold-out test set 비율 (0이면 test set 없이 K-fold CV만 수행)")
    parser.add_argument("--experiment_name", default=None, type=str,
                        help="실험 이름 (결과 파일명에 사용)")
    parser.add_argument("--save_fold_models", action="store_true",
                        help="각 폴드별 모델을 개별 파일로 저장 (out-of-sample 분석용)")
    parser.add_argument("--save_intermediates", action="store_true",
                        help="Out-of-sample inference 중간 산물 저장 (pipeline 분석용)")

    # wandb 설정
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="NLP-ASD-Korean", type=str)

    args = parser.parse_args()

    # Config 생성
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'patience': args.patience,
        'ff_dropout': args.ff_dropout,
        'att_dropout': args.att_dropout,
        'class_dropout': args.class_dropout,
        'sentence_dropout': args.sentence_dropout,
        'weight_decay': args.weight_decay,
        'k_fold_CV': args.k_fold_CV,
        'inner_val_splits': args.inner_val_splits,
        'model_name': args.model_name,
        'report_max_length': args.report_max_length,
        'sentence_max_length': args.sentence_max_length,
        'data_mode': args.data_mode,
        'meta_csv_path': args.meta_csv_path,
        'tokenized_path': args.tokenized_path,
        'seed': args.seed,
        'test_ratio': args.test_ratio,
        'experiment_name': args.experiment_name,
        'save_fold_models': args.save_fold_models,
        'save_intermediates': args.save_intermediates,
        'wandb_project': args.wandb_project,
    }

    print("=" * 60)
    print("NLP-ASD Korean Model Training (v4.1)")
    print("=" * 60)
    print(f"\n설정: {config}\n")

    # 학습 실행
    results = train_model(config, use_wandb=not args.no_wandb, verbose=True)

    # 결과 출력
    print("\n" + "=" * 60)
    print("=== 최종 결과 (평균 +/- 표준편차) ===")
    print("=" * 60)

    summary = results['summary']
    print(f"AUROC:       {summary['avg_auroc']:.4f} (+/- {summary['std_auroc']:.4f})")
    print(f"AUPRC:       {summary['avg_auprc']:.4f} (+/- {summary['std_auprc']:.4f})")
    print(f"Accuracy:    {summary['avg_accuracy']:.4f} (+/- {summary['std_accuracy']:.4f})")
    print(f"Precision:   {summary['avg_precision']:.4f} (+/- {summary['std_precision']:.4f})")
    print(f"Recall:      {summary['avg_recall']:.4f} (+/- {summary['std_recall']:.4f})")
    print(f"Specificity: {summary['avg_specificity']:.4f} (+/- {summary['std_specificity']:.4f})")
    print(f"F1:          {summary['avg_f1']:.4f} (+/- {summary['std_f1']:.4f})")
    print(f"Loss:        {summary['avg_loss']:.4f} (+/- {summary['std_loss']:.4f})")

    print("\n--- Fold별 결과 ---")
    for r in results['fold_results']:
        m = r['metrics']
        print(f"Fold {r['fold']} (Best Epoch={r['best_epoch']}, Inner-valid AUROC={r['inner_valid_auroc']:.4f}): "
              f"Outer AUROC={m['auroc']:.4f}, Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 파일명 생성 (experiment_name이 있으면 사용)
    if config.get('experiment_name'):
        results_file = os.path.join(results_dir, f"{config['experiment_name']}.txt")
    else:
        results_file = os.path.join(results_dir, f"training_results_v4_1_{timestamp}.txt")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("NLP-ASD Korean Model Training Results (v4.1)\n")
        f.write("=" * 60 + "\n\n")

        f.write("### 설정 ###\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

        f.write("\n### CV 결과 (평균 +/- 표준편차) ###\n")
        f.write(f"AUROC:       {summary['avg_auroc']:.4f} (+/- {summary['std_auroc']:.4f})\n")
        f.write(f"AUPRC:       {summary['avg_auprc']:.4f} (+/- {summary['std_auprc']:.4f})\n")
        f.write(f"Accuracy:    {summary['avg_accuracy']:.4f} (+/- {summary['std_accuracy']:.4f})\n")
        f.write(f"Precision:   {summary['avg_precision']:.4f} (+/- {summary['std_precision']:.4f})\n")
        f.write(f"Recall:      {summary['avg_recall']:.4f} (+/- {summary['std_recall']:.4f})\n")
        f.write(f"Specificity: {summary['avg_specificity']:.4f} (+/- {summary['std_specificity']:.4f})\n")
        f.write(f"F1:          {summary['avg_f1']:.4f} (+/- {summary['std_f1']:.4f})\n")
        f.write(f"Loss:        {summary['avg_loss']:.4f} (+/- {summary['std_loss']:.4f})\n")

        # Hold-out Test 결과 (있는 경우)
        if results.get('test_metrics'):
            test_m = results['test_metrics']
            f.write("\n### Hold-out Test 결과 ###\n")
            f.write(f"Test AUROC:       {test_m['auroc']:.4f}\n")
            f.write(f"Test AUPRC:       {test_m['auprc']:.4f}\n")
            f.write(f"Test Accuracy:    {test_m['accuracy']:.4f}\n")
            f.write(f"Test Precision:   {test_m['precision']:.4f}\n")
            f.write(f"Test Recall:      {test_m['recall']:.4f}\n")
            f.write(f"Test Specificity: {test_m['specificity']:.4f}\n")
            f.write(f"Test F1:          {test_m['f1']:.4f}\n")

        f.write("\n### Fold별 결과 ###\n")
        for r in results['fold_results']:
            m = r['metrics']
            f.write(f"Fold {r['fold']} (Best Epoch={r['best_epoch']}, Inner-valid AUROC={r['inner_valid_auroc']:.4f}): "
                    f"AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, "
                    f"Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, "
                    f"Rec={m['recall']:.4f}, Spec={m['specificity']:.4f}, F1={m['f1']:.4f}\n")

        # Fold별 클래스 분포 및 Confusion Matrix 추가
        f.write("\n### Fold별 클래스 분포 및 Confusion Matrix ###\n")
        splits = results['splits']
        label_tensor_cv = results['label_tensor_cv']

        total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0

        for fold_idx, (train_idx, valid_idx) in enumerate(splits):
            r = results['fold_results'][fold_idx]
            m = r['metrics']

            # 클래스 분포 계산
            train_labels = label_tensor_cv[train_idx]
            valid_labels = label_tensor_cv[valid_idx]
            train_pos = int(train_labels.sum().item())
            train_neg = len(train_labels) - train_pos
            valid_pos = int(valid_labels.sum().item())
            valid_neg = len(valid_labels) - valid_pos

            # Confusion Matrix
            TP, FP, TN, FN = m['TP'], m['FP'], m['TN'], m['FN']
            total_TP += TP
            total_FP += FP
            total_TN += TN
            total_FN += FN

            f.write(f"\nFold {fold_idx + 1}:\n")
            f.write(f"  Train: {len(train_idx)} samples (Pos={train_pos}, Neg={train_neg}, Pos비율={train_pos/len(train_idx):.1%})\n")
            f.write(f"  Valid: {len(valid_idx)} samples (Pos={valid_pos}, Neg={valid_neg}, Pos비율={valid_pos/len(valid_idx):.1%})\n")
            f.write(f"  Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}\n")

            # 예측 패턴 경고
            if TP == 0 and FP == 0:
                f.write(f"  [Warning] 모든 샘플을 Negative로 예측\n")
            elif TN == 0 and FN == 0:
                f.write(f"  [Warning] 모든 샘플을 Positive로 예측\n")

        # 전체 합계
        f.write(f"\n전체 Confusion Matrix 합계:\n")
        f.write(f"  TP={total_TP}, FP={total_FP}, TN={total_TN}, FN={total_FN}\n")
        if total_TP + total_FP > 0:
            f.write(f"  Overall Precision: {total_TP / (total_TP + total_FP):.4f}\n")
        if total_TP + total_FN > 0:
            f.write(f"  Overall Recall: {total_TP / (total_TP + total_FN):.4f}\n")
        if total_TN + total_FP > 0:
            f.write(f"  Overall Specificity: {total_TN / (total_TN + total_FP):.4f}\n")
        total_samples = total_TP + total_FP + total_TN + total_FN
        if total_samples > 0:
            f.write(f"  Overall Accuracy: {(total_TP + total_TN) / total_samples:.4f}\n")

        f.write(f"\n저장 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n결과 저장: {results_file}")

    # 모델 저장 (test_ratio > 0이고 best_model_state가 있는 경우)
    if config.get('test_ratio', 0) > 0 and results.get('best_model_state'):
        model_dir = os.path.join(ROOT, "trained_models")
        os.makedirs(model_dir, exist_ok=True)
        if config.get('experiment_name'):
            model_file = os.path.join(model_dir, f"{config['experiment_name']}.pt")
        else:
            model_file = os.path.join(model_dir, f"best_model_{timestamp}.pt")
        torch.save(results['best_model_state'], model_file)
        print(f"모델 저장: {model_file}")

    # Out-of-sample inference 중간 산물 저장
    if results.get('oos_results') is not None:
        exp_name = config.get('experiment_name', f'experiment_{timestamp}')
        intermediates_dir = os.path.join(ROOT, "intermediates", exp_name)
        os.makedirs(intermediates_dir, exist_ok=True)

        oos = results['oos_results']
        np.save(os.path.join(intermediates_dir, "logits_np.npy"), oos['logits_np'])
        np.save(os.path.join(intermediates_dir, "probs_np.npy"), oos['probs_np'])
        np.save(os.path.join(intermediates_dir, "predicted_labels_np.npy"), oos["predicted_labels_np"])
        np.save(os.path.join(intermediates_dir, "attention_matrices_np.npy"), oos['attention_matrices_np'])
        np.save(os.path.join(intermediates_dir, "attention_weighted_sentence_embs_np.npy"), oos['attention_weighted_sentence_embs_np'])
        np.save(os.path.join(intermediates_dir, "lhs_embs_np.npy"), oos['lhs_embs_np'])
        np.save(os.path.join(intermediates_dir, "layer_pooled_embs_np.npy"), oos['layer_pooled_embs_np'])
        np.save(os.path.join(intermediates_dir, "labels_np.npy"), oos['labels_np'])
        np.save(os.path.join(intermediates_dir, "fold_assignment.npy"), oos['fold_assignment'])
        np.save(os.path.join(intermediates_dir, "report_id_array.npy"), oos["report_id_array"])
        np.save(os.path.join(intermediates_dir, "patient_id_array.npy"), oos["patient_id_array"])
        np.save(os.path.join(intermediates_dir, "cv_original_indices.npy"), oos["cv_original_indices"])
        np.save(os.path.join(intermediates_dir, "valid_sentence_mask_np.npy"), oos["valid_sentence_mask_np"])
        np.save(os.path.join(intermediates_dir, "n_valid_sentences_np.npy"), oos["n_valid_sentences_np"])

        with open(os.path.join(intermediates_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        with open(os.path.join(intermediates_dir, "fold_split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(results["fold_split_manifest"], f, ensure_ascii=False, indent=2)

        with open(os.path.join(intermediates_dir, "fold_checkpoint_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(results["fold_checkpoint_manifest"], f, ensure_ascii=False, indent=2)

        print(f"\n중간 산물 저장: {intermediates_dir}")
        print(f"  - logits_np.npy: {oos['logits_np'].shape}")
        print(f"  - probs_np.npy: {oos['probs_np'].shape}")
        print(f"  - attention_matrices_np.npy: {oos['attention_matrices_np'].shape}")
        print(f"  - attention_weighted_sentence_embs_np.npy: {oos['attention_weighted_sentence_embs_np'].shape}")
        print(f"  - lhs_embs_np.npy: {oos['lhs_embs_np'].shape}")
        print(f"  - layer_pooled_embs_np.npy: {oos['layer_pooled_embs_np'].shape}")
        print(f"  - labels_np.npy: {oos['labels_np'].shape}")
        print(f"  - fold_assignment.npy: {oos['fold_assignment'].shape}")

    print("\n학습 완료!")


if __name__ == "__main__":
    main()
