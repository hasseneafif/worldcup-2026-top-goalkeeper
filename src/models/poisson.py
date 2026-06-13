"""Stage 1 match model: Elo-driven Poisson with Dixon-Coles correction.

Expected goals are driven by the pre-match Elo difference (the strongest single
predictor) rather than per-team attack/defense MLE, so the model applies cleanly to
any of the 48 teams and to matchups that have never occurred:

    log(lambda_home) = mu + home_adv * (not neutral) + beta * elo_diff/100
    log(lambda_away) = mu                            - beta * elo_diff/100

The Dixon-Coles tau term corrects the independent-Poisson under-prediction of
0-0/1-0/0-1/1-1 scorelines. Fit by weighted MLE (time-decay x importance weights).
Set use_dc=False to get the plain independent-Poisson baseline for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

MODELS = Path(__file__).resolve().parents[2] / "outputs"
ELO_SCALE = 100.0
MAX_GOALS = 10


def _pois_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lam = np.clip(lam, 1e-9, None)
    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _dc_tau(x, y, lh, la, rho):
    """Dixon-Coles low-score correction (vectorized over arrays of x,y,lh,la)."""
    tau = np.ones_like(lh, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    tau[m01] = 1.0 + lh[m01] * rho
    tau[m10] = 1.0 + la[m10] * rho
    tau[m11] = 1.0 - rho
    return np.clip(tau, 1e-9, None)


@dataclass
class DixonColesElo:
    mu: float = 0.2
    home_adv: float = 0.3
    beta: float = 0.6
    rho: float = -0.05
    use_dc: bool = True

    # ---- core lambda model ----
    def lambdas(self, elo_diff, neutral):
        elo_diff = np.asarray(elo_diff, dtype=float) / ELO_SCALE
        neutral = np.asarray(neutral, dtype=bool)
        ha = np.where(neutral, 0.0, self.home_adv)
        lh = np.exp(self.mu + ha + self.beta * elo_diff)
        la = np.exp(self.mu - self.beta * elo_diff)
        return lh, la

    # ---- fitting ----
    def fit(self, df, weights=None):
        x = df["home_score"].to_numpy(dtype=float)
        y = df["away_score"].to_numpy(dtype=float)
        elo_diff = df["elo_diff"].to_numpy(dtype=float)
        neutral = df["neutral"].to_numpy(dtype=bool)
        w = np.ones(len(df)) if weights is None else np.asarray(weights, dtype=float)

        def nll(theta):
            mu, home_adv, beta, rho = theta
            ed = elo_diff / ELO_SCALE
            ha = np.where(neutral, 0.0, home_adv)
            lh = np.exp(mu + ha + beta * ed)
            la = np.exp(mu - beta * ed)
            ll = _pois_logpmf(x, lh) + _pois_logpmf(y, la)
            if self.use_dc:
                ll = ll + np.log(_dc_tau(x, y, lh, la, rho))
            return -np.sum(w * ll)

        x0 = [self.mu, self.home_adv, self.beta, self.rho]
        bounds = [(-1.0, 1.5), (0.0, 1.0), (0.0, 2.0), (-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        self.mu, self.home_adv, self.beta, self.rho = res.x
        self._nll = res.fun
        return self

    # ---- prediction ----
    def score_matrix(self, lh: float, la: float) -> np.ndarray:
        gh = np.arange(MAX_GOALS + 1)
        ph = np.exp(_pois_logpmf(gh.astype(float), np.full(MAX_GOALS + 1, lh)))
        pa = np.exp(_pois_logpmf(gh.astype(float), np.full(MAX_GOALS + 1, la)))
        mat = np.outer(ph, pa)
        if self.use_dc:
            # apply tau to the four corrected cells
            for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                tau = _dc_tau(np.array([i]), np.array([j]), np.array([lh]), np.array([la]), self.rho)[0]
                mat[i, j] *= tau
        return mat / mat.sum()

    def outcome_probs(self, elo_diff: float, neutral: bool) -> tuple[float, float, float]:
        """Return (P_home_win, P_draw, P_away_win) for one matchup."""
        lh, la = self.lambdas(np.array([elo_diff]), np.array([neutral]))
        mat = self.score_matrix(float(lh[0]), float(la[0]))
        p_home = np.tril(mat, -1).sum()  # home goals > away goals
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        return float(p_home), float(p_draw), float(p_away)

    def outcome_probs_batch(self, elo_diff, neutral) -> np.ndarray:
        """(N,3) array of [P_home, P_draw, P_away]; loops the score matrix per row."""
        elo_diff = np.asarray(elo_diff, dtype=float)
        neutral = np.asarray(neutral, dtype=bool)
        out = np.empty((len(elo_diff), 3))
        for i in range(len(elo_diff)):
            out[i] = self.outcome_probs(elo_diff[i], bool(neutral[i]))
        return out

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {"mu": self.mu, "home_adv": self.home_adv, "beta": self.beta,
                "rho": self.rho, "use_dc": self.use_dc}

    @classmethod
    def from_dict(cls, d: dict) -> "DixonColesElo":
        return cls(**d)
