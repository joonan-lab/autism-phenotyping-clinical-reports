# -*- coding: utf-8 -*-
"""Shared helpers for the report-type confound-controlled (type-residual) pipeline.

모든 step 스크립트가 이 모듈을 import 해서 동일한 입력 로딩 / 잔차화(residualization) /
PCA / GMM / t-분포 / Mardia tail 계산을 공유한다. 입력 데이터 위치는 DATA_DIR 한 곳에서만
지정한다.

Residualization = 각 ILR 좌표에서 보고서 유형(A vs P, report_id 끝의 첫 글자) 지표를 OLS 로
regress-out. 선형연산이라 정규직교 기저변환과 교환되므로 PCA/GMM/Mahalanobis/eta2 가 기저 불변.

도메인 개수는 메타(JSON)의 domain_columns 에서 읽어 data-driven 으로 처리한다(18/19 무관).
"""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.special import digamma, gammaln

# ── 입력 데이터 (도메인 빈도 벡터) ───────────────────────────────────────────
# 번들 외부의 도메인 빈도 벡터 산출물을 가리킨다. 다른 데이터로 돌리려면 이 경로만 바꾼다.
DATA_DIR = paths.VECTORS
PROPORTION_TSV = DATA_DIR / "domain_vectors_proportion_latest.tsv"
META_JSON = DATA_DIR / "domain_vectors_meta_latest.json"

SEED = 42
STABILITY_THRESHOLD = 0.8
PSEUDOCOUNT = 1e-6


def helmert(D):
    V = np.zeros((D, D - 1))
    for i in range(D - 1):
        V[:i + 1, i] = 1.0 / (i + 1)
        V[i + 1, i] = -1.0
        V[:, i] *= np.sqrt((i + 1.0) / (i + 2.0))
    return V


def clr(X):
    Xp = X + PSEUDOCOUNT
    Xp = Xp / Xp.sum(1, keepdims=True)
    L = np.log(Xp)
    return L - L.mean(1, keepdims=True)


def ilr(X):
    return clr(X) @ helmert(X.shape[1])


def residualize(M, design):
    R = np.zeros_like(M)
    for j in range(M.shape[1]):
        b = np.linalg.lstsq(design, M[:, j], rcond=None)[0]
        R[:, j] = M[:, j] - design @ b
    return R


def variance_partition(X, labels):
    g = X.mean(0)
    total = ((X - g) ** 2).sum()
    between = sum(np.sum(labels == k) * ((X[labels == k].mean(0) - g) ** 2).sum()
                 for k in np.unique(labels))
    return total, between, total - between, between / total


def load():
    """Return the canonical type-residual representation + raw helpers (ASD only)."""
    df = pd.read_csv(PROPORTION_TSV, sep="\t")
    dom = json.load(open(META_JSON))["domain_columns"]
    sub = df[df["asd_label"].values == 1].reset_index(drop=True)
    rid = sub["report_id"].values
    P = np.load(DATA_DIR / "domain_vectors_proportion.npy")[df["asd_label"].values == 1]
    rtype = np.array([r.split("-")[-1][0] for r in rid])
    design = np.column_stack([np.ones(len(P)), (rtype == "P").astype(float)])
    ILR = ilr(P)
    RES = residualize(ILR, design)
    CLR_RES = residualize(clr(P), design)
    return dict(rid=rid, dom=dom, P=P, rtype=rtype,
                n_A=int((rtype == "A").sum()), n_P=int((rtype == "P").sum()),
                ILR=ILR, RES=RES, CLR_RES=CLR_RES, helmert=helmert(P.shape[1]))


def pca_scores(M, k_lead=3):
    p = PCA(n_components=M.shape[1], random_state=SEED).fit(M)
    s = p.transform(M)
    return p, s, s[:, :k_lead], p.explained_variance_ratio_


def gmm_fit(lead, k):
    g = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                        random_state=SEED, max_iter=500).fit(lead)
    return g.predict(lead), g.predict_proba(lead), g


# ── heavy-tail / multivariate-t helpers (step3 GMM-vs-t, step4 tail) ──────────
def mardia_z(X):
    """Multivariate kurtosis (Mardia) z-score + 99th-percentile excess (quantile ratio)."""
    n, p = X.shape
    mu = X.mean(0)
    ci = np.linalg.inv(np.cov(X, rowvar=False))
    d2 = np.sum((X - mu) @ ci * (X - mu), 1)
    beta2 = np.mean(d2 ** 2)
    exp = p * (p + 2)
    z = (beta2 - exp) / np.sqrt(8 * p * (p + 2) / n)
    chi99 = stats.chi2.ppf(0.99, df=p)
    exc = np.percentile(d2, 99) / chi99
    return dict(p=int(p), beta2=float(beta2), expected=int(exp), z=float(z),
                excess99=float(exc), kurt="lepto" if z > 0 else "platy")


def t_loglik(X, mu, S, nu):
    n, p = X.shape
    d = X - mu
    _, ld = np.linalg.slogdet(S)
    Si = np.linalg.inv(S)
    m2 = np.sum(d @ Si * d, 1)
    return np.sum(gammaln((nu + p) / 2) - gammaln(nu / 2) - (p / 2) * np.log(nu * np.pi)
                  - 0.5 * ld - ((nu + p) / 2) * np.log(1 + m2 / nu))


def fit_t(X):
    """EM fit of a multivariate-t. Returns (nu, loglik, n_params)."""
    n, p = X.shape
    mu = X.mean(0)
    S = np.cov(X, rowvar=False)
    nu = 5.0
    prev = -np.inf
    for _ in range(300):
        d = X - mu
        Si = np.linalg.inv(S)
        m2 = np.sum(d @ Si * d, 1)
        w = (nu + p) / (nu + m2)
        ws = w.sum()
        mu = (w[:, None] * X).sum(0) / ws
        dn = X - mu
        S = (dn * w[:, None]).T @ dn / n
        S = (S + S.T) / 2
        if np.linalg.eigvalsh(S).min() < 1e-10:
            S += np.eye(p) * 1e-6

        def eq(v):
            return (-digamma(v / 2) + np.log(v / 2) + 1 + (1 / n) * np.sum(np.log(w) - w)
                    + digamma((v + p) / 2) - np.log((v + p) / 2))
        lo, hi = 0.1, 500.0
        for _ in range(100):
            m = (lo + hi) / 2
            if eq(m) > 0:
                lo = m
            else:
                hi = m
            if hi - lo < 0.01:
                break
        nu = (lo + hi) / 2
        ll = t_loglik(X, mu, S, nu)
        if abs(ll - prev) < 1e-6:
            break
        prev = ll
    return nu, ll, p + p * (p + 1) // 2 + 1
