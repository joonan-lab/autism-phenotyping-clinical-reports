"""
한국어용 Sentence Attention BERT 모델 (v1.3.1)

v1.3.1 변경사항 (from v1.2):
- Padding 문장에 대한 attention masking 추가
- key_padding_mask를 사용하여 padding 문장에 attention이 부여되지 않도록 수정
- 문장 수준 마스크 자동 생성 기능 추가

원본: sentence_attention_base_pool_ko_v1_2.py
"""

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig

# ============================================================================
# 사용 가능한 한국어 Pre-trained 모델 목록
# ============================================================================
#
# 권장 모델 (KLUE):
#   - "klue/roberta-base"      : 가장 범용적, 논문 재현에 적합 (권장)
#   - "klue/roberta-large"     : 더 높은 성능, 더 많은 GPU 메모리 필요
#   - "klue/bert-base"         : BERT 아키텍처 선호 시
#
# 기타 옵션:
#   - "monologg/koelectra-base-v3-discriminator" : 효율적인 학습
#   - "beomi/kcbert-base"      : 구어체/댓글 데이터에 강점
#
# 의료 도메인:
#   - 현재 공개된 한국어 의료 특화 모델은 제한적
#   - 일반 모델로 시작 후 추가 사전학습 권장
# ============================================================================


class SelfAttentionAverage(nn.Module):
    """
    Single-head Self-Attention으로 문장들을 가중 평균

    v1.3.1 변경사항:
    - key_padding_mask 지원 추가로 padding 문장 마스킹
    """

    def __init__(self, embed_dim=768, weight_dim=768, dropout=0.0, classifier_dropout=0.0):
        """
        Args:
            embed_dim (int): 입력 임베딩 차원 (KLUE RoBERTa base = 768)
            weight_dim (int): attention weight 차원
            dropout (float): attention dropout
            classifier_dropout (float): 분류기 dropout
        """
        super(SelfAttentionAverage, self).__init__()

        # Single-head attention (논문의 핵심 - 해석 가능성을 위해 1개 헤드만 사용)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=1,  # 단일 헤드 - 해석 가능성 확보
            kdim=weight_dim,
            vdim=weight_dim,
            batch_first=True,
            dropout=dropout
        )

        self.dropout_layer = nn.Dropout(p=classifier_dropout)

        # 최종 이진 분류 레이어
        self.linear = nn.Linear(weight_dim, 1)

    def forward(self, x, attn_mask=None, key_padding_mask=None, average_attn_weights=False):
        """
        Args:
            x: 문장 임베딩 (batch, num_sentences, embed_dim)
            attn_mask: 어텐션 마스크 (additive mask for attention weights)
            key_padding_mask: padding 마스크 (batch, num_sentences) - True인 위치가 padding
            average_attn_weights: 어텐션 가중치 평균 여부

        Returns:
            log_reg: 로짓 (batch,)
            attn_weights: 어텐션 가중치 (batch, num_sentences, num_sentences)
            attn_output: 어텐션 적용된 출력 (batch, num_sentences, embed_dim)
        """
        attn_output, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            average_attn_weights=average_attn_weights
        )

        # key_padding_mask가 있으면 padding 문장의 출력을 0으로 설정
        if key_padding_mask is not None:
            # key_padding_mask: (batch, num_sentences) - True인 위치가 padding
            mask_expanded = key_padding_mask.unsqueeze(-1)  # (batch, num_sentences, 1)
            attn_output = attn_output.masked_fill(mask_expanded, 0.0)

        # 유효한 문장만 평균 계산
        if key_padding_mask is not None:
            # 유효한 문장 수 계산
            valid_mask = ~key_padding_mask  # True인 위치가 유효한 문장
            valid_count = valid_mask.sum(dim=1, keepdim=True).float()  # (batch, 1)
            valid_count = valid_count.clamp(min=1.0)  # 0으로 나누는 것 방지

            # 유효한 문장들만 합산 후 평균
            pooled_output = attn_output.sum(dim=1) / valid_count.squeeze(-1).unsqueeze(-1)  # (batch, embed_dim)
        else:
            # 모든 문장의 평균 (attention-weighted)
            pooled_output = torch.mean(attn_output, dim=1)

        pooled_output_drop = self.dropout_layer(pooled_output)

        # 이진 분류 (batch 차원 유지를 위해 squeeze(-1) 사용)
        log_reg = self.linear(pooled_output_drop).squeeze(-1)

        return log_reg, attn_weights, attn_output


class SentenceAttentionBERTKorean(nn.Module):
    """
    한국어용 Sentence-level Attention BERT 모델 (v1.3.1)

    v1.3.1 변경사항:
    - Padding 문장에 대한 attention masking 자동 적용
    - 토큰 수준 attention mask를 문장 수준으로 변환하는 기능 추가
    """

    def __init__(
        self,
        base_model_name="klue/roberta-base",  # 한국어 모델로 변경
        sentence_embed_dim=768,
        sentence_weight_dim=768,
        word_embed_dim=768,
        word_weight_dim=768,
        sentence_dropout=0.0,
        ff_dropout=0.1,
        att_dropout=0.1,
        report_max_length=64,
        class_dropout=0.0
    ):
        """
        Args:
            base_model_name (str): HuggingFace 모델 이름
            sentence_embed_dim (int): 문장 임베딩 차원
            sentence_weight_dim (int): 문장 attention weight 차원
            ff_dropout (float): Feed-forward dropout
            att_dropout (float): Attention dropout
            report_max_length (int): 보고서당 최대 문장 수
            class_dropout (float): 분류기 dropout
        """
        super(SentenceAttentionBERTKorean, self).__init__()

        # 모델 설정 로드
        config = AutoConfig.from_pretrained(base_model_name)

        # 임베딩 차원 자동 감지 (모델마다 다를 수 있음)
        self.hidden_size = config.hidden_size  # 보통 768

        # Dropout 설정을 config에 적용
        config.hidden_dropout_prob = ff_dropout
        config.attention_probs_dropout_prob = att_dropout

        # num_labels를 config에 설정
        config.num_labels = sentence_embed_dim

        # 베이스 모델 로드 (분류 헤드 포함)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            config=config,
            ignore_mismatched_sizes=True
        )

        # 문장 수준 Self-Attention
        self.sentence_attention = SelfAttentionAverage(
            embed_dim=sentence_embed_dim,
            weight_dim=sentence_weight_dim,
            dropout=sentence_dropout,
            classifier_dropout=class_dropout
        )

        self.report_max_length = report_max_length

    def _create_sentence_padding_mask(self, attn_mask):
        """
        토큰 수준 attention mask에서 문장 수준 padding mask 생성

        Args:
            attn_mask: (batch, num_sentences, seq_length) - 토큰 수준 마스크

        Returns:
            sentence_padding_mask: (batch, num_sentences) - True인 위치가 padding 문장
        """
        # 각 문장의 유효한 토큰 수 계산
        # attn_mask가 1인 곳이 유효한 토큰
        # 문장이 전부 padding이면 (모든 토큰이 0) 해당 문장은 padding
        valid_token_count = attn_mask.sum(dim=-1)  # (batch, num_sentences)

        # 유효한 토큰이 0개인 문장 = padding 문장
        # 또는 유효한 토큰이 매우 적은 경우 (special tokens만 있는 경우)
        # [CLS], [SEP] 등 special token만 있으면 실제 내용이 없는 문장
        # 일반적으로 유효 토큰이 2개 이하면 padding으로 취급 (CLS, SEP만 있는 경우)
        sentence_padding_mask = valid_token_count <= 2

        return sentence_padding_mask

    def forward(self, input, attn_mask=None, sentence_attn_mask=None):
        """
        Forward pass

        Args:
            input: 입력 텐서 (batch, num_sentences, seq_length)
            attn_mask: 토큰 수준 어텐션 마스크 (batch, num_sentences, seq_length)
            sentence_attn_mask: 문장 수준 어텐션 마스크 (deprecated, 자동 생성됨)

        Returns:
            logits: 분류 로짓 (batch,)
            sentence_attn_weights: 문장 어텐션 가중치
            attn_output: 어텐션 출력
            last_hidden_embs: 마지막 레이어 임베딩
            layer_pooled_embs: 레이어별 pooled 임베딩
        """
        # 문장 수준 padding mask 생성
        if attn_mask is not None:
            sentence_padding_mask = self._create_sentence_padding_mask(attn_mask)
        else:
            sentence_padding_mask = None

        batch_cls = []
        batch_hidden_states = []
        batch_layer_pooled = []

        for report_ind in range(input.shape[0]):
            report_cls = []
            report_hidden_states = []

            # 마이크로 배치 처리 (메모리 효율성)
            for micro_batch_inds in range(int(input.shape[1] / self.report_max_length)):
                start_idx = micro_batch_inds * self.report_max_length
                end_idx = (micro_batch_inds + 1) * self.report_max_length

                single_report_input = input[report_ind, start_idx:end_idx, :].squeeze()
                single_report_mask = attn_mask[report_ind, start_idx:end_idx, :].squeeze() if attn_mask is not None else None

                # BERT로 각 문장 인코딩
                base_outputs = self.base_model(
                    single_report_input,
                    attention_mask=single_report_mask,
                    output_hidden_states=True
                )

                # 문장 임베딩
                base_cls = base_outputs.logits

                # hidden states 추출
                hidden_states = base_outputs.hidden_states

                report_cls.append(base_cls)
                report_hidden_states.append(hidden_states)

            report_cls = torch.cat(report_cls, dim=0)
            batch_cls.append(report_cls)

            # 각 레이어별로 hidden states 병합
            num_layers = len(report_hidden_states[0])
            merged_hidden_full = []
            layer_pooled_list = []

            for layer_idx in range(num_layers):
                layer_tensors = [hs[layer_idx] for hs in report_hidden_states]
                merged_layer = torch.cat(layer_tensors, dim=0)

                merged_hidden_full.append(merged_layer)

                pooled = merged_layer.mean(dim=0).mean(dim=0)
                layer_pooled_list.append(pooled)

            batch_hidden_states.append(merged_hidden_full)
            batch_layer_pooled.append(torch.stack(layer_pooled_list, dim=0))

        batch_cls = torch.stack(batch_cls, dim=0)

        # 마지막 레이어 임베딩
        last_hidden_embs = [hs[-1] for hs in batch_hidden_states]

        # 모든 레이어 임베딩
        layer_pooled_embs = batch_layer_pooled

        # 문장 수준 Attention 적용 (padding mask 전달)
        logits, sentence_attn_weights, attn_output = self.sentence_attention(
            batch_cls,
            attn_mask=sentence_attn_mask,
            key_padding_mask=sentence_padding_mask
        )

        return logits, sentence_attn_weights, attn_output, last_hidden_embs, layer_pooled_embs


# 별칭 (원본 코드와의 호환성)
SentenceAttentionBERT = SentenceAttentionBERTKorean
