# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:19:22 2025

@author: Guillaume
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def GED(R,
        S,
        covR: np.ndarray | None = None,
        covS: np.ndarray | None = None,
        n_comp='all',
        reg=True):
    
    # covR = np.cov(R)
    # covS = np.cov(S)
    if covR is None:
        covR = R@R.T/(R.shape[1]-1)
        covR = 0.5 * (covR + covR.T)
    if covS is None:
        covS = S@S.T/(S.shape[1]-1)
        covS = 0.5 * (covS + covS.T)
    
    if reg:
        covR = regularize(covR)
        covS = regularize(covS)
    
    ## Source separation via GED
    
    # GED
    D,W = sp.linalg.eigh(covS,covR)
    
    # sort eigenvalues/vectors
    sidx  = np.flip(np.argsort(D))
    D = D[sidx]
    W = W[:,sidx]

    # Eigenvector sign
    se = np.max(np.abs(W),axis=0,keepdims=True)
    W = W*np.sign(se)
    
    W_norm = W/np.linalg.norm(W,axis=0,keepdims=True)
    
    if type(n_comp) is not str:
        if n_comp<1.0:
            idx = np.where(np.cumsum(D/np.sum(D))<n_comp)[0]
        else:
            idx = np.arange(n_comp)
    else:
        idx = np.arange(D.size)
    
    W = W[:,idx]
    W_norm = W_norm[:,idx]
    
    A = W.T@covS
    
    var_exp = D[idx]/np.sum(D)
    
    return W,A,W_norm,var_exp
    

def regularize(cov, lam=1e-6, min_eps=None):
    """
    cov: (n,n) covariance (assumed symmetric or will be symmetrized outside)
    lam: dimensionless loading; effective ridge = lam * trace(cov)/n
    min_eps: optional absolute floor for eps to avoid zero (e.g., 1e-12)
    """
    cov = sym(cov)
    n = cov.shape[0]
    mean_var = np.trace(cov) / n
    # Guard: if mean_var <= 0 (numerical), fall back to tiny positive
    if not np.isfinite(mean_var) or mean_var <= 0:
        mean_var = 1.0
    eps = lam * mean_var
    if min_eps is not None:
        eps = max(eps, min_eps)
    return cov + eps * np.eye(n)

def sym(M): return 0.5*(M+M.T)

def rho_ratio(w,R,S):
    """Rayleigh ratio on given w."""
    den = np.float64(w.T @ R @ w)
    num = np.float64(w.T @ S @ w)
    if w.ndim>1:
        rho_ratio = np.diag(num)/np.diag(den)
        rho_ratio[rho_ratio<=0] = np.nan
    else:
        rho_ratio = num/den
        if rho_ratio<=0:
            rho_ratio = np.nan
    return rho_ratio

def align_to_template(U_i, U_t):
    # U: (n_chan, n_comp), template: (n_chan,)
    t = U_t / (np.linalg.norm(U_t) + 1e-12)
    sims = U_i.T @ t  # cosine up to norms; U columns assumed unit-ish
    k = int(np.argmax(np.abs(sims)))
    u = U_i[:, k]*np.sign(sims[k])
    
    return u, k, float(np.abs(sims[k]))

def random_rho_distribution(R, S, n_samples=500, rng=None):
    rng = np.random.default_rng(rng)
    n = S.shape[0]
    rhos = []
    for _ in range(n_samples):
        w = rng.standard_normal(n)
        w /= np.linalg.norm(w)
        rhos.append(rho_ratio(w,R,S))
    return np.array(rhos)

def random_rho_shuffling(R, S, W, n_samples=500, rng=None):
    rng = np.random.default_rng(rng)
    n = S.shape[0]
    rhos = []
    for _ in range(n_samples):
        r_i = np.random.randint(0, high=n, size=(n))
        w = np.array([W[r_i[iC],iC] for iC in range(n)])
        # w = rng.standard_normal(n)
        w /= np.linalg.norm(w)
        rhos.append(rho_ratio(w,R,S))
    return np.array(rhos)

def _group_GED(covRs, covSs, eps=1e-6, normalize='trace'):
    """
    covRs, covSs: lists of (n_chan, n_chan) covariances per subject ()
    Returns:
        evecs  : (n_chan, n_comp) group eigenvectors in whitened space
        evals: (n_comp,) eigenvalues
        Ws : list of per-subject filters (n_chan, n_comp), back-transformed
        As : list of per-subject forward models (n_chan, n_comp)
    """
    n_subj = len(covRs)
    n = covRs[0].shape[0]
    Ls = []
    covS_whits = []

    covR_reg = []
    for covR, covS in zip(covRs, covSs):
        covR = sym(covR)
        covS = sym(covS)
        covR = regularize(covR,lam=eps)
        covR_reg.append(covR)
            
        # Cholesky; if it fails, increase eps or use eig-based whitening.
        L = sp.linalg.cholesky(covR, lower=True)
        Linv = np.linalg.solve(L, np.eye(n))      # L^{-1}
        covS_whit = Linv @ covS @ Linv.T               # whitened cov R
        if normalize == 'trace':
            tr = np.trace(covS_whit)
            if tr > 0: covS_whit /= tr
        elif normalize == 'fro':
            fn = np.linalg.norm(covS_whit, 'fro')
            if fn > 0: covS_whit /= fn
        Ls.append(L)
        covS_whits.append(covS_whit)

    covS_bar = sum(covS_whits) / n_subj

    # Standard eigen in whitened space
    evals, evecs = np.linalg.eigh(covS_bar)                # ascending
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    # Back-transform per subject and B-normalize
    Ws = []
    As = []
    for iS in range(n_subj):
        L = Ls[iS]
        LinvT = np.linalg.solve(L.T, np.eye(n))   # L^{-T}
        W_i = LinvT @ evecs                            # filters in subject space
        # B-normalize each column: w^T B w = 1
        covR = covR_reg[iS]
        covS = sym(covSs[iS])
        for k in range(W_i.shape[1]):
            denom = np.sqrt(W_i[:,k].T @ covR @ W_i[:,k])
            if denom > 0: W_i[:,k] /= denom
        Ws.append(W_i)
        
        A_i = covS @ W_i @ np.linalg.inv(W_i.T @ covS @ W_i)
        As.append(A_i)
    
    # tmp_A = sum(As)/len(As)
    # for iS in range(n_subj):
    #     np.dot(As[iS],tmp_A)
        
    return evecs, evals, Ws, As  # U is group template in whitened space; Ws are per-subject filters

def group_GED(covRs, covSs, eps=1e-6, normalize='trace', CV=True):
    """
    LOSO CV: for each subject j as test, train metric-aware group GED on others,
    back-transform filters for j, and evaluate held-out rho on (A_j, B_j).
    Returns:
      rho_mat: (n_subj, K) held-out ratios for the top-K components
    """
    
    U, D, Ws, As = _group_GED(covRs, covSs, eps=eps, normalize=normalize)
    
    K = covSs[0].shape[0]
    n_subj = len(covSs)
    n_rand = 1000
    
    rho_mat = np.full((n_subj, K), np.nan)
    z_rho = np.full((n_subj, K), np.nan)
    rho_rand = np.full((n_subj, n_rand), np.nan)
    for j in range(n_subj):
        
        W_norm = Ws[j]/np.linalg.norm(Ws[j],axis=0)
        
        rho_mat[j, :] = rho_ratio(W_norm,sym(covRs[j]),sym(covSs[j]))
        rho_rand = np.log(random_rho_shuffling(sym(covRs[j]),sym(covSs[j]),
                                               Ws[j],
                                              n_samples=n_rand, rng=None))
        # rho_rand = np.log(random_rho_distribution(sym(covRs[j]),sym(covSs[j]),
        #                                           n_samples=n_rand, rng=None))

        z_rho[j,:] = (np.log(rho_mat[j, :])-np.nanmean(rho_rand))/np.nanstd(rho_rand)
        

    # rhos_rand = random_rho_distribution(S_eval, R_eval)
    # print("random ρ mean ~", rhos_rand.mean())
    
    # if CV:
    #     K = covSs[0].shape[0]
        
    #     n_subj = len(covSs)
    #     rho_mat = np.full((n_subj, K), np.nan)
    
    
    #     for j in range(n_subj):
    #         tr_idx = [i for i in range(n_subj) if i != j]
    #         U_tr, _, _, _ = _group_GED([covRs[i] for i in tr_idx],
    #                                   [covSs[i] for i in tr_idx],
    #                                   eps=eps, normalize=normalize)
    #         for k in range(K):
    #             # align one component to the template
    #             u_best, k_best, sim = align_to_template(U_tr, U[:,k])
        
    #             # back-transform to subject j using their B_j
    #             B_reg = regularize(sym(covRs[j]), lam=eps)
    #             L_j = sp.linalg.cholesky(B_reg, lower=True)
    #             w_j = np.linalg.solve(L_j.T, u_best)       # (n,)
    #             # R-normalize
    #             d = np.sqrt(w_j.T @ B_reg @ w_j);  w_j = w_j / (d if d>0 else 1.0)
        
    #             # held-out rho on **unregularized** R
    #             rho_mat[j, k] = rho_ratio(w_j, sym(covSs[j]), sym(covRs[j]))

    return U, D, Ws, As, z_rho

    

def spoc(
    X,
    z,
    *,
    n_bootstrapping_iterations: int = 0,
    pca_X_var_explained: float = 1.0,
    verbose: int = 1,
    Cxx: np.ndarray | None = None,
    Cxxz: np.ndarray | None = None,
    Cxxe: np.ndarray | None = None,
    random_state: int | np.random.Generator | None = None,
):
    """
    Source Power Co-modulation Analysis (SPoC).
    Python translation of the MATLAB function by S. Dähne (2014).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_epochs)
        Band-pass filtered, epoched data. If Cxx/Cxxz/Cxxe are provided you
        may pass X=None.
    z : ndarray, shape (n_epochs,)
        Univariate target variable (will be z-scored).
    n_bootstrapping_iterations : int, default=0
        If >0, build a null distribution of lambda via phase-randomized
        surrogates of z and return p-values.
    pca_X_var_explained : float in (0, 1], default=1.0
        Optional dimensionality reduction via PCA using cumulative variance threshold.
        1.0 disables dimensionality reduction.
    verbose : int, default=1
        If >0, prints occasional progress during bootstrapping.
    Cxx, Cxxz, Cxxe : optional precomputed matrices
        Cxx : (n_channels, n_channels), grand-average covariance
        Cxxe: (n_channels, n_channels, n_epochs), mean-free per-epoch covariances
        Cxxz: (n_channels, n_channels), z-weighted covariance
        If provided, they override computation from X.
    random_state : int | np.random.Generator | None
        RNG for bootstrapping. If int, used to seed a Generator.

    Returns
    -------
    W : ndarray, shape (n_channels, n_components)
        Spatial filters (columns), ordered by descending lambda.
    A : ndarray, shape (n_channels, n_components)
        Spatial patterns (columns), aligned with W.
    lambda_values : ndarray, shape (n_components,)
        Covariances between projected power and target function z.
    p_values_lambda : ndarray, shape (n_components,)
        One-sided p-values from the bootstrap (NaN if bootstrapping disabled).
    Cxx : ndarray, shape (n_channels, n_channels)
        Grand-average covariance.
    Cxxz : ndarray, shape (n_channels, n_channels)
        z-weighted covariance.
    Cxxe : ndarray, shape (n_channels, n_channels, n_epochs)
        Mean-free per-epoch covariances.
    """

    # --- input checks & shapes
    if X is None and (Cxx is None or Cxxe is None):
        raise ValueError("If X is None, you must provide both Cxx and Cxxe.")

    if X is not None:
        n_samples, n_channels, n_epochs = X.shape
    else:
        _, n_channels, n_epochs = Cxxe.shape

    z = np.asarray(z).reshape(-1)
    if len(z) != n_epochs:
        raise ValueError("X and z must have the same number of epochs.")

    # --- normalize target (zero mean, unit variance)
    z = sp.stats.zscore(z, ddof=0)

    # --- compute covariance structures if not provided
    if Cxx is None:
        Cxx = _compute_Cxx_from_X(X)
    if Cxxe is None:
        Cxxe = _compute_Cxxe_from_X(X, Cxx)
    if Cxxz is None:
        Cxxz = _create_Cxxz(Cxxe, z)

    # --- whitening (with optional PCA dimension selection)
    M, keep_idx, evals, evecs = _whiten_from_Cxx(Cxx, pca_X_var_explained)
    # In whitened space, generalized eigenproblem reduces to ordinary one
    Cxxz_white = M @ Cxxz @ M.T

    # --- eigendecomposition in whitened space
    # eigh returns sorted ascending; we'll sort ourselves by descending
    d, W_white = np.linalg.eigh(Cxxz_white)
    idx = np.argsort(d)[::-1]
    lambda_values = d[idx]
    W_white = W_white[:, idx]

    # --- back-project filters to sensor space
    # W_white columns live in whitened PCA subspace; project via M^T
    W = (M.T) @ W_white

    # --- unit-variance normalization of filters: w^T Cxx w = 1
    for k in range(W.shape[1]):
        denom = float(W[:, [k]].T @ Cxx @ W[:, [k]])
        if denom <= 0:
            # numerical safety; skip scaling if degenerate
            continue
        W[:, k] /= np.sqrt(denom + 1e-15)

    # --- patterns (forward models): A = Cxx W (W^T Cxx W)^{-1}
    G = W.T @ Cxx @ W
    # numerical stabilization
    G = 0.5 * (G + G.T)
    G_inv = np.linalg.pinv(G)
    A = Cxx @ W @ G_inv

    # --- optional bootstrap for p-values (phase-randomized surrogates of z)
    if n_bootstrapping_iterations > 0:
        rng = _resolve_rng(random_state)
        lambda_samples = np.empty(n_bootstrapping_iterations, dtype=float)
        # Precompute FFT amplitude of z for faster surrogates
        z_amps = None
        for b in range(n_bootstrapping_iterations):
            if verbose and (b + 1) % max(1, n_bootstrapping_iterations // 25) == 0:
                print(f"bootstrapping iteration {b+1}/{n_bootstrapping_iterations}")

            z_shuffled, z_amps = _random_phase_surrogate_1d(z, z_amps, rng)
            Cxxz_s = _create_Cxxz(Cxxe, z_shuffled)
            Cxxz_s_white = M @ Cxxz_s @ M.T
            d_s, _ = np.linalg.eigh(Cxxz_s_white)
            lambda_samples[b] = np.max(np.abs(d_s))

        # one-sided (greater or equal) on absolute values, matching MATLAB
        p_values_lambda = np.array([
            np.mean(np.abs(lambda_samples) >= np.abs(lam)) for lam in lambda_values
        ])
    else:
        p_values_lambda = np.full_like(lambda_values, np.nan, dtype=float)

    return W, A, lambda_values, p_values_lambda, Cxx, Cxxz, Cxxe


# ------------------------ helper functions ------------------------ #

def _compute_Cxx_from_X(X: np.ndarray) -> np.ndarray:
    """Grand-average covariance across epochs."""
    _, n_channels, n_epochs = X.shape
    Cxx = np.zeros((n_channels, n_channels), dtype=float)
    for e in range(n_epochs):
        X_e = X[:, :, e]
        # numpy.cov expects variables in columns when rowvar=False
        Cxx += np.cov(X_e, rowvar=False, bias=False)
    Cxx /= n_epochs
    # symmetrize for numerical hygiene
    return 0.5 * (Cxx + Cxx.T)


def _compute_Cxxe_from_X(X: np.ndarray, Cxx: np.ndarray) -> np.ndarray:
    """Mean-free per-epoch covariance matrices."""
    _, n_channels, n_epochs = X.shape
    Cxxe = np.zeros((n_channels, n_channels, n_epochs), dtype=float)
    for e in range(n_epochs):
        X_e = X[:, :, e]
        C_tmp = np.cov(X_e, rowvar=False, bias=False)
        Cxxe[:, :, e] = C_tmp - Cxx   # mean-free (matches MATLAB code)
    return Cxxe


def _create_Cxxz(Cxxe: np.ndarray, z: np.ndarray) -> np.ndarray:
    """z-weighted covariance: average over epochs of z(e) * Cxxe_e."""
    # tensordot over epoch dimension
    Cxxz = np.tensordot(Cxxe, z, axes=([2], [0])) / Cxxe.shape[2]
    return 0.5 * (Cxxz + Cxxz.T)  # enforce symmetry


def _whiten_from_Cxx(Cxx: np.ndarray, var_thresh: float):
    """
    Compute whitening transform M such that M @ Cxx @ M.T = I (on kept subspace),
    using PCA/eigendecomposition. Optionally drop trailing components by
    cumulative variance threshold.
    """
    # symmetric eigendecomposition
    evals, evecs = np.linalg.eigh(Cxx)
    # sort descending by variance
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # guard against tiny/negative eigenvalues (numerical)
    evals = np.clip(evals, a_min=1e-15, a_max=None)

    # choose number of PCs by cumulative explained variance
    if not (0 < var_thresh <= 1):
        raise ValueError("pca_X_var_explained must be in (0, 1].")
    cumsum = np.cumsum(evals) / np.sum(evals)
    k = np.searchsorted(cumsum, var_thresh, side="left") + 1
    k = min(k, len(evals))

    evals_k = evals[:k]
    evecs_k = evecs[:, :k]

    # whitening transform: M = Λ^{-1/2} E^T  (so that M Cxx M^T = I)
    M = (evecs_k.T) / np.sqrt(evals_k)[None, :]
    return M, np.arange(k), evals_k, evecs_k


def _random_phase_surrogate_1d(z: np.ndarray, precomputed_amps: np.ndarray | None,
                               rng: np.random.Generator):
    """
    Phase-randomized surrogate for a 1D real signal z.
    Preserves amplitude spectrum; draws i.i.d. phases for positive frequencies
    (excluding DC and Nyquist), then enforces Hermitian symmetry.
    Returns (z_surrogate, amplitude_spectrum) to reuse spectrum across iterations.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    Z = np.fft.rfft(z)
    if precomputed_amps is None:
        amps = np.abs(Z)
    else:
        amps = precomputed_amps

    # random phases for bins 1..(Nfft-2) when length>2; keep DC (0) and Nyquist (last) real
    n_bins = Z.size
    phases = np.zeros(n_bins, dtype=float)
    if n_bins > 2:
        phases[1:-1] = rng.uniform(0, 2 * np.pi, size=n_bins - 2)
    # DC and Nyquist phases remain zero

    Zs = amps * np.exp(1j * phases)
    z_sur = np.fft.irfft(Zs, n=n).real
    # re-normalize to match mean/variance of original z (like zscore & inverse)
    z_sur = (z_sur - z_sur.mean()) / (z_sur.std(ddof=0) + 1e-15)
    return z_sur, amps


def _resolve_rng(random_state):
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))
    return np.random.default_rng()