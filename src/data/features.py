"""Training weights and supplementary features for the match model.

Two weights multiply into the likelihood so recent, competitive matches dominate:
  * time decay  — exponential, configurable half-life (default ~18 months)
  * importance  — friendlies count less than competitive fixtures
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# importance weight by tournament keyword (relative; friendlies are downweighted)
IMPORTANCE = [
    ("FIFA World Cup qualification", 1.0),
    ("FIFA World Cup", 1.0),
    ("Confederations Cup", 0.9),
    ("UEFA Euro", 0.9),
    ("Copa América", 0.9),
    ("African Cup of Nations", 0.85),
    ("AFC Asian Cup", 0.85),
    ("Gold Cup", 0.8),
    ("Nations League", 0.8),
    ("qualification", 0.85),
    ("Friendly", 0.5),
]
DEFAULT_IMPORTANCE = 0.7


def importance_weight(tournament: str) -> float:
    t = str(tournament)
    for kw, w in IMPORTANCE:
        if kw.lower() in t.lower():
            return w
    return DEFAULT_IMPORTANCE


def time_decay_weight(dates: pd.Series, ref_date: pd.Timestamp, half_life_days: float = 547.0) -> np.ndarray:
    """Exponential decay; weight = 0.5 ** (age_in_days / half_life). Default half-life ~18 months."""
    age = (ref_date - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    return 0.5 ** (age / half_life_days)


def add_form_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add leakage-free as-of recent-form features via one chronological pass.

    For each match, both teams' rolling stats use only their *prior* matches:
      *_form  : average points per game over last `window` matches (3/1/0)
      *_gf/ga : average goals scored / conceded over last `window` matches
    Columns: home_form, away_form, home_gf, home_ga, away_gf, away_ga, plus
    diff features form_diff, gf_diff, ga_diff. Requires df sorted by date.
    """
    from collections import defaultdict, deque

    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    cols = {k: np.full(len(df), np.nan) for k in
            ["home_form", "away_form", "home_gf", "home_ga", "away_gf", "away_ga"]}

    def stats(team):
        h = hist[team]
        if not h:
            return 1.0, 1.2, 1.2  # neutral priors: ~1 pt/game, ~1.2 goals
        pts = np.mean([p for p, _, _ in h])
        gf = np.mean([f for _, f, _ in h])
        ga = np.mean([a for _, _, a in h])
        return pts, gf, ga

    for i, row in enumerate(df.itertuples(index=False)):
        hp, hgf, hga = stats(row.home_team)
        ap, agf, aga = stats(row.away_team)
        cols["home_form"][i], cols["home_gf"][i], cols["home_ga"][i] = hp, hgf, hga
        cols["away_form"][i], cols["away_gf"][i], cols["away_ga"][i] = ap, agf, aga
        # update history with this match's outcome
        if row.home_score > row.away_score:
            ph, pa = 3, 0
        elif row.home_score < row.away_score:
            ph, pa = 0, 3
        else:
            ph, pa = 1, 1
        hist[row.home_team].append((ph, row.home_score, row.away_score))
        hist[row.away_team].append((pa, row.away_score, row.home_score))

    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    out["form_diff"] = out["home_form"] - out["away_form"]
    out["gf_diff"] = out["home_gf"] - out["away_gf"]
    out["ga_diff"] = out["home_ga"] - out["away_ga"]
    return out


def recent_match_counts(df: pd.DataFrame, ref_date: pd.Timestamp | None = None,
                        window_days: int = 730) -> dict[str, int]:
    """Matches each team played within `window_days` before ref_date.

    Used to size per-team rating uncertainty: fewer recent matches -> noisier Elo.
    """
    ref_date = ref_date or df["date"].max()
    recent = df[(df["date"] >= ref_date - pd.Timedelta(days=window_days)) & (df["date"] <= ref_date)]
    counts: dict[str, int] = {}
    for col in ("home_team", "away_team"):
        for team, c in recent[col].value_counts().items():
            counts[team] = counts.get(team, 0) + int(c)
    return counts


def add_weights(
    df: pd.DataFrame,
    ref_date: pd.Timestamp | None = None,
    half_life_days: float = 547.0,
) -> pd.DataFrame:
    """Add `w_time`, `w_importance`, and combined `weight` columns.

    ref_date defaults to the latest date in df. For a leakage-free backtest of
    tournament T, pass ref_date = T's start and pre-filter df to dates < that.
    """
    out = df.copy()
    ref_date = ref_date or out["date"].max()
    out["w_time"] = time_decay_weight(out["date"], ref_date, half_life_days)
    out["w_importance"] = out["tournament"].map(importance_weight)
    out["weight"] = out["w_time"] * out["w_importance"]
    return out
