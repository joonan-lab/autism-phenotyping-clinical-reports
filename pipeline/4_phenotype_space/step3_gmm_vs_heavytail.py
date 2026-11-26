"""Observed GMM and unimodal fits after type and length adjustment."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma,gammaln
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import _common_controlled as rn
SEED=42
GMM_SEEDS=tuple(range(10))
N_INIT=10
MAX_ITER=100
K_MAX=8
NU_STARTS=(2.,5.,30.)
PSEUDOCOUNTS=(1e-6,1e-4,1e-2)
HERE=Path(__file__).resolve().parent
def ilr_coords(P, pc):
    X = np.asarray(P, float) + pc
    X = X / X.sum(1, keepdims=True)
    L = np.log(X)
    return (L - L.mean(1, keepdims=True)) @ rn.helmert(X.shape[1])


def design_matrix(name, types, n):
    isP = (types == "P").astype(float)
    ln = np.log(n.astype(float))
    cols = [np.ones(len(n)), isP]
    if name in ("TL", "TLx"):
        cols.append(ln)
    if name == "TLx":
        cols.append(isP * ln)
    return np.column_stack(cols)


def lead_scores(P, types, n, design, pc):
    ilr = ilr_coords(P, pc)
    D = design_matrix(design, types, n)
    R = ilr - D @ np.linalg.lstsq(D, ilr, rcond=None)[0]
    pca = PCA(n_components=R.shape[1], svd_solver="full", random_state=SEED).fit(R)
    S = pca.transform(R)
    return S[:, :3], pca.explained_variance_ratio_


def gmm_bics(X):
    """Best BIC per k over GMM_SEEDS; also returns labels of the BIC-optimal fit."""
    best = {}
    for k in range(1, K_MAX + 1):
        b_k, g_k = np.inf, None
        for s in GMM_SEEDS:
            g = GaussianMixture(n_components=k, covariance_type="full", n_init=N_INIT,
                                max_iter=MAX_ITER, random_state=s).fit(X)
            b = g.bic(X)
            if b < b_k:
                b_k, g_k = b, g
        best[k] = (float(b_k), g_k)
    kbest = min(best, key=lambda k: best[k][0])
    return {k: v[0] for k, v in best.items()}, kbest, best[kbest][1].predict(X)


def t_loglik(X, mu, S, nu):
    n, p = X.shape
    d = X - mu
    _, ld = np.linalg.slogdet(S)
    m2 = np.sum(d @ np.linalg.inv(S) * d, 1)
    return np.sum(gammaln((nu + p) / 2) - gammaln(nu / 2) - (p / 2) * np.log(nu * np.pi)
                  - 0.5 * ld - ((nu + p) / 2) * np.log(1 + m2 / nu))


def fit_t(X, nu0):
    """EM fit of a multivariate t; same algorithm as 6.1 _common_controlled.fit_t with a chosen nu start."""
    n, p = X.shape
    mu = X.mean(0)
    S = np.cov(X, rowvar=False)
    nu = float(nu0)
    prev = -np.inf
    ll = prev
    for _ in range(300):
        d = X - mu
        m2 = np.sum(d @ np.linalg.inv(S) * d, 1)
        w = (nu + p) / (nu + m2)
        mu = (w[:, None] * X).sum(0) / w.sum()
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
    n_par = p + p * (p + 1) // 2 + 1
    return float(nu), float(ll), float(-2 * ll + n_par * np.log(n))


def best_t(X):
    fits = [fit_t(X, v) for v in NU_STARTS]
    nu, ll, bic = max(fits, key=lambda f: f[1])
    return {"t_nu": nu, "t_loglik": ll, "t_bic": bic, "t_nu_at_bound": bool(nu > 499.0)}


def mardia(X):
    n, p = X.shape
    Xc = X - X.mean(0)
    S = np.cov(X, rowvar=False, bias=True)
    G = Xc @ np.linalg.solve(S, Xc.T)
    skew_stat = n * float((G ** 3).mean()) / 6
    d2 = np.diag(G)
    beta2 = float(np.mean(d2 ** 2))
    kurt_z = (beta2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    return skew_stat, float(kurt_z)


def cramers_v(a, b):
    tab = pd.crosstab(a, b).values
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    r, c = tab.shape
    return float(np.sqrt(chi2 / (tab.sum() * (min(r, c) - 1))))


def analyse(tag, P, types, n, design, pc, keep_labels=False):
    X, evr = lead_scores(P, types, n, design, pc)
    gb, kbest, labels = gmm_bics(X)
    t = best_t(X)
    skew, kz = mardia(X)
    row = {"dataset": tag, "design": design, "pseudocount": pc, "n_reports": len(n),
           **{f"gmm_bic_k{k}": gb[k] for k in gb}, "gmm_best_k": kbest, **t,
           "pc1_3_variance": float(evr[:3].sum()), "mardia_skew_stat": skew, "mardia_kurt_z": kz,
           "zero_rate_A": float((P[types == "A"] == 0).mean()), "zero_rate_P": float((P[types == "P"] == 0).mean())}
    return row, X, (labels if keep_labels else None)


def run_skewt(lead_by_tag, out_dir):
    csv = out_dir / "skewt_input.csv"
    tsv = out_dir / "skewt_fits.tsv"
    frames = [pd.DataFrame(X, columns=["PC1", "PC2", "PC3"]).assign(dataset=tag) for tag, X in lead_by_tag.items()]
    pd.concat(frames)[["dataset", "PC1", "PC2", "PC3"]].to_csv(csv, index=False)
    subprocess.run(["Rscript", str(HERE / "fit_skewt.R"), str(csv), str(tsv)], check=True)
    csv.unlink()
    return pd.read_csv(tsv, sep="\t")


def add_dbic(df):
    df["gauss_bic"] = df["gmm_bic_k1"]
    df["unimodal_best_bic"] = df[["gauss_bic", "t_bic", "skewt_bic"]].min(axis=1, skipna=True)
    df["unimodal_best_model"] = df[["gauss_bic", "t_bic", "skewt_bic"]].idxmin(axis=1, skipna=True).str.replace("_bic", "")
    for k in range(1, K_MAX + 1):
        df[f"dbic_k{k}"] = df[f"gmm_bic_k{k}"] - df["unimodal_best_bic"]
    df["dbic_min"] = df[[f"dbic_k{k}" for k in range(1, K_MAX + 1)]].min(axis=1)
    df["dbic_min_k"] = df[[f"dbic_k{k}" for k in range(1, K_MAX + 1)]].idxmin(axis=1).str.replace("dbic_k", "").astype(int)
    return df


def main():
 d=rn.load();ids=np.load(paths.INTER/'report_id_array.npy',allow_pickle=False).astype(str);mask=np.load(paths.INTER/'valid_sentence_mask_np.npy',allow_pickle=False);lookup={r:i for i,r in enumerate(ids)};n=mask[[lookup[r] for r in d['rid']]].sum(1)
 rows=[];leads={}
 for pc in PSEUDOCOUNTS:
  for design in ['T','TL']:
   tag=f'obs_{design}_pc{pc:g}';row,X,_=analyse(tag,d['P'],d['rtype'],n,design,pc);rows.append(row);leads[tag]=X
 tag='obs_TLx_pc1e-06';row,X,_=analyse(tag,d['P'],d['rtype'],n,'TLx',1e-6);rows.append(row);leads[tag]=X
 out=paths.out(paths.SPACE);skew=run_skewt(leads,out);skew['skewt_bic']=skew[['st_bic','sn_bic']].min(axis=1);df=add_dbic(pd.DataFrame(rows).merge(skew,on='dataset',how='left'));df.to_csv(out/'observed_bic.tsv',sep='\t',index=False)
if __name__=='__main__':main()
