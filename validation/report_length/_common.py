"""Type-adjusted coordinates and tail summaries for report-length matching."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
SEED=42
PSEUDOCOUNT=1e-6
TAIL_QUANTILE=.9
def helmert(D):
    V = np.zeros((D, D - 1))
    for i in range(D - 1):
        V[:i + 1, i] = 1 / (i + 1)
        V[i + 1, i] = -1
        V[:, i] *= np.sqrt((i + 1) / (i + 2))
    return V


def coordinate_fit(P, types, pseudocount=PSEUDOCOUNT, decimals=None):
    """pseudocount 뒤 재정규화, Helmert ILR, 절편+P형 OLS 잔차, PCA 18축(svd_solver full)."""
    X = np.round(P, decimals) if decimals is not None else np.array(P, dtype=float)
    X = X + pseudocount
    X /= X.sum(1, keepdims=True)
    L = np.log(X)
    ilr = (L - L.mean(1, keepdims=True)) @ helmert(X.shape[1])
    design = np.column_stack([np.ones(len(X)), (types == 'P').astype(float)])
    R = ilr - design @ np.linalg.lstsq(design, ilr, rcond=None)[0]
    if np.linalg.matrix_rank(R) != R.shape[1]:
        raise np.linalg.LinAlgError('rank-deficient residual coordinates')
    pca = PCA(n_components=R.shape[1], svd_solver='full', random_state=SEED).fit(R)
    return R, pca.transform(R), pca.explained_variance_ratio_


def mahalanobis_d2(X):
    """표본 평균과 공분산(ddof 1)으로 계산한 Mahalanobis d². 정칙화하지 않는다."""
    d = X - X.mean(0)
    cov = np.cov(X, rowvar=False)
    if np.linalg.matrix_rank(cov) != X.shape[1]:
        raise np.linalg.LinAlgError('singular covariance')
    d2 = np.einsum('ij,ij->i', d, np.linalg.solve(cov, d.T).T)
    if not (np.isfinite(d2).all() and d2.min() > -1e-8):
        raise np.linalg.LinAlgError('invalid Mahalanobis distances')
    return d2


def tail_metrics(X):
    n, p = X.shape
    d2 = mahalanobis_d2(X)
    beta = float(np.mean(d2 ** 2))
    ref = float(stats.chi2.ppf(0.99, p))
    return {'beta2': beta, 'z': (beta - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n),
            'q99_d2_ratio': float(np.quantile(d2, 0.99) / ref),
            'fraction_above_chi99': float(np.mean(d2 > ref))}, d2


def analyze(P, types, lengths=None, pseudocount=PSEUDOCOUNT, decimals=None):
    """f(P). 거리는 18차원 잔차 공간 Mahalanobis이고 꼬리는 d ≥ q0.9다."""
    R, scores, evr = coordinate_fit(P, types, pseudocount, decimals)
    out = {}
    dist = None
    for name, X in (('full', R), ('leading', scores[:, :3]), ('residual', scores[:, 3:])):
        m, d2 = tail_metrics(X)
        out.update({f'{name}_{k}': v for k, v in m.items()})
        if name == 'full':
            dist = np.sqrt(np.clip(d2, 0, None))
    Pz = np.round(P, decimals) if decimals is not None else P
    is_a, is_p = types == 'A', types == 'P'
    tail = dist >= np.quantile(dist, TAIL_QUANTILE)
    rho = np.nan
    if lengths is not None and len(np.unique(lengths)) > 1:
        rho = float(stats.spearmanr(dist, lengths).statistic)
    out.update({
        'effective_dim': float(np.exp(-np.sum(evr * np.log(evr)))),
        'pc1_variance': float(evr[0]), 'leading_variance': float(evr[:3].sum()),
        'overall_zero_rate': float(np.mean(Pz == 0)),
        'zero_rate_A': float(np.mean(Pz[is_a] == 0)), 'zero_rate_P': float(np.mean(Pz[is_p] == 0)),
        'mean_nonzero_domains': float(np.mean(np.sum(Pz > 0, 1))),
        'tail_n': int(tail.sum()), 'tail_A': int((tail & is_a).sum()), 'tail_P': int((tail & is_p).sum()),
        'tail_A_fraction': float((tail & is_a).sum() / is_a.sum()),
        'tail_P_fraction': float((tail & is_p).sum() / is_p.sum()),
        'tail_P_share': float((tail & is_p).sum() / tail.sum()),
        'rho_distance_length': rho,
    })
    return out, dist


def seed_for(phase, choice, rep):
    return int(np.random.SeedSequence([SEED, phase, choice, rep]).generate_state(1, dtype=np.uint64)[0])
