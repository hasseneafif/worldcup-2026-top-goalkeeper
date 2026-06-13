"""Self-computed international Elo (eloratings.net style).

One chronological pass over results assigns every match the *pre-match* Elo of both
teams (so it is leakage-free by construction) and leaves a dict of current ratings.
Computing our own Elo from the same results.csv we train on avoids cross-source
team-name mismatches entirely, and gives full control of as-of-date logic.

Formula:
    E_home = 1 / (1 + 10 ** (-(R_home + HFA*not_neutral - R_away) / 400))
    R' = R + K * G * (W - E)
where W in {1, 0.5, 0}, G is a goal-difference multiplier, and K scales with match
importance (World Cup > continental > qualifier > friendly).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0  # added to the non-neutral home side

# match-importance K factor by tournament keyword (checked in order)
K_BY_KEYWORD = [
    ("FIFA World Cup qualification", 40),
    ("FIFA World Cup", 60),
    ("Confederations Cup", 50),
    ("UEFA Euro qualification", 40),
    ("UEFA Euro", 50),
    ("Copa América", 50),
    ("African Cup of Nations qualification", 40),
    ("African Cup of Nations", 50),
    ("AFC Asian Cup qualification", 40),
    ("AFC Asian Cup", 50),
    ("Gold Cup", 50),
    ("UEFA Nations League", 45),
    ("CONCACAF Nations League", 45),
    ("qualification", 40),
    ("Friendly", 20),
]
DEFAULT_K = 30


def k_factor(tournament: str) -> int:
    t = str(tournament)
    for kw, k in K_BY_KEYWORD:
        if kw.lower() in t.lower():
            return k
    return DEFAULT_K


def goal_diff_multiplier(diff: int) -> float:
    """eloratings.net goal-difference weighting."""
    d = abs(int(diff))
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    if d == 3:
        return 1.75
    return 1.75 + (d - 3) / 8.0


def expected_score(r_home: float, r_away: float, neutral: bool) -> float:
    adj = 0.0 if neutral else HOME_ADVANTAGE
    return 1.0 / (1.0 + 10 ** (-(r_home + adj - r_away) / 400.0))


def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add home_elo_pre / away_elo_pre columns; return (df, final_ratings)."""
    ratings: dict[str, float] = {}
    home_pre = np.empty(len(df))
    away_pre = np.empty(len(df))

    for i, row in enumerate(df.itertuples(index=False)):
        rh = ratings.get(row.home_team, BASE_RATING)
        ra = ratings.get(row.away_team, BASE_RATING)
        home_pre[i] = rh
        away_pre[i] = ra

        eh = expected_score(rh, ra, row.neutral)
        if row.home_score > row.away_score:
            wh = 1.0
        elif row.home_score < row.away_score:
            wh = 0.0
        else:
            wh = 0.5

        k = k_factor(row.tournament) * goal_diff_multiplier(row.home_score - row.away_score)
        delta = k * (wh - eh)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta

    out = df.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    return out, ratings


def main() -> None:
    df = pd.read_pickle(INTERIM / "results_clean.pkl")
    out, ratings = compute_elo(df)
    out.to_pickle(INTERIM / "results_elo.pkl")
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:20]
    print("results with as-of Elo ->", INTERIM / "results_elo.pkl")
    print("\nTop 20 current Elo (as of last match in data):")
    for team, r in top:
        print(f"  {team:25s} {r:7.1f}")


if __name__ == "__main__":
    main()
