"""Golden Glove Monte Carlo — best-goalkeeper probabilities for WC 2026.

The Golden Glove is driven by clean sheets, which in turn come from team defensive
strength and how many matches the team plays. So we reuse the tournament simulator
and, in every match, record each team's goals conceded and whether it kept a clean
sheet (a 0-0 won on penalties still counts as a clean sheet for both keepers). Over
many tournaments, the keeper most likely to top the clean-sheet chart wins.

Odds are computed at team level (defense x depth); each team's first-choice keeper
(keepers.py) is attached as the label.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean import canonical
from src.data.features import add_weights
from src.data.keepers import keeper_for
from src.models.elo import compute_elo
from src.models.poisson import DixonColesElo
from src.simulate.bracket import (
    ALL_TEAMS, FINAL, GROUPS, HOST_ELO_BONUS, KO_TREE, QUARTERS, R16, R32, SEMIS,
    group_fixtures,
)

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"


def _bipartite_match(slot_allowed, group_of):
    n = len(slot_allowed)
    match_team = [-1] * n

    def try_assign(team_k, seen):
        for s in range(n):
            if group_of[team_k] in slot_allowed[s] and not seen[s]:
                seen[s] = True
                if match_team[s] == -1 or try_assign(match_team[s], seen):
                    match_team[s] = team_k
                    return True
        return False

    for k in range(len(group_of)):
        if not try_assign(k, [False] * n):
            return None
    return match_team


def fit_match_model() -> DixonColesElo:
    df = pd.read_pickle(INTERIM / "results_elo.pkl")
    df = add_weights(df, ref_date=df["date"].max())
    return DixonColesElo(use_dc=True).fit(df, weights=df["weight"].to_numpy())


class GoldenGloveSimulator:
    def __init__(self, model, ratings):
        self.model = model
        self.teams = ALL_TEAMS
        self.idx = {t: i for i, t in enumerate(self.teams)}
        elo = np.array([ratings[t] for t in self.teams], dtype=float)
        for t, bonus in HOST_ELO_BONUS.items():
            elo[self.idx[t]] += bonus
        self.elo = elo
        self.group_idx = {g: [self.idx[t] for t in teams] for g, teams in GROUPS.items()}
        self.third_slots = [(m, sa[1]) for (m, _, sa) in R32 if sa[0] == "3"]

    def _lambdas(self, i, j):
        ed = (self.elo[i] - self.elo[j]) / 100.0
        return np.exp(self.model.mu + self.model.beta * ed), np.exp(self.model.mu - self.model.beta * ed)

    def _match(self, i, j, rng, cs, ga):
        lh, la = self._lambdas(i, j)
        gi, gj = rng.poisson(lh), rng.poisson(la)
        ga[i] += gj; ga[j] += gi
        if gj == 0:
            cs[i] += 1           # team i kept a clean sheet
        if gi == 0:
            cs[j] += 1
        return gi, gj

    def _knockout_winner(self, i, j, rng, cs, ga):
        gi, gj = self._match(i, j, rng, cs, ga)
        if gi > gj:
            return i
        if gj > gi:
            return j
        p_i = 1.0 / (1.0 + 10 ** (-(self.elo[i] - self.elo[j]) / 400.0))
        return i if rng.random() < p_i else j

    def _simulate_group(self, idxs, rng, cs, ga):
        pts = np.zeros(4); gf = np.zeros(4); gag = np.zeros(4)
        for a, b in group_fixtures(idxs):
            ga_, gb_ = self._match(idxs[a], idxs[b], rng, cs, ga)
            gf[a] += ga_; gag[a] += gb_; gf[b] += gb_; gag[b] += ga_
            if ga_ > gb_:
                pts[a] += 3
            elif gb_ > ga_:
                pts[b] += 3
            else:
                pts[a] += 1; pts[b] += 1
        key = pts * 1e6 + (gf - gag) * 1e3 + gf + rng.random(4) * 1e-3
        order = np.argsort(-key)
        return [idxs[o] for o in order], pts, (gf - gag), gf, order

    def simulate_once(self, rng):
        n = len(self.teams)
        cs = np.zeros(n, dtype=np.int32)   # clean sheets
        ga = np.zeros(n, dtype=np.int32)   # goals against
        winners, runners, thirds = {}, {}, []
        for g, idxs in self.group_idx.items():
            ranked, pts, gd, gf, order = self._simulate_group(idxs, rng, cs, ga)
            winners[g], runners[g] = ranked[0], ranked[1]
            o3 = order[2]
            thirds.append((g, ranked[2], pts[o3], gd[o3], gf[o3]))

        thirds.sort(key=lambda x: (x[2], x[3], x[4], rng.random()), reverse=True)
        best_thirds = thirds[:8]
        slot_allowed = [set(letters) for (_m, letters) in self.third_slots]
        assign = _bipartite_match(slot_allowed, [g for (g, *_r) in best_thirds]) or list(range(8))
        third_by_match = {self.third_slots[s][0]: best_thirds[k][1] for s, k in enumerate(assign)}

        win_by_match = {}
        for (m, sh, sa) in R32:
            h = self._slot_team(sh, winners, runners, third_by_match, m)
            a = third_by_match[m] if sa[0] == "3" else self._slot_team(sa, winners, runners, third_by_match, m)
            win_by_match[m] = self._knockout_winner(h, a, rng, cs, ga)
        for rnd in (R16, QUARTERS, SEMIS):
            for m in rnd:
                f1, f2 = KO_TREE[m]
                win_by_match[m] = self._knockout_winner(win_by_match[f1], win_by_match[f2], rng, cs, ga)
        f1, f2 = KO_TREE[FINAL]
        self._knockout_winner(win_by_match[f1], win_by_match[f2], rng, cs, ga)

        # Golden Glove winner: most clean sheets, then fewest goals conceded
        key = cs * 1000.0 - ga + rng.random(n) * 1e-3
        return int(np.argmax(key)), cs, ga

    def _slot_team(self, slot, winners, runners, third_by_match, match_no):
        kind, ref = slot
        if kind == "W":
            return winners[ref]
        if kind == "R":
            return runners[ref]
        return third_by_match[match_no]

    def run(self, n_sims: int, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = len(self.teams)
        wins = np.zeros(n); cs_sum = np.zeros(n); ga_sum = np.zeros(n); top3 = np.zeros(n)
        for _ in range(n_sims):
            w, cs, ga = self.simulate_once(rng)
            wins[w] += 1
            cs_sum += cs; ga_sum += ga
            top3[np.argsort(-(cs * 1000.0 - ga))[:3]] += 1
        df = pd.DataFrame({
            "team": self.teams,
            "keeper": [keeper_for(t) for t in self.teams],
            "P_golden_glove": wins / n_sims,
            "exp_clean_sheets": cs_sum / n_sims,
            "exp_conceded": ga_sum / n_sims,
            "P_top3": top3 / n_sims,
        })
        return df.sort_values("P_golden_glove", ascending=False).reset_index(drop=True)


def main(n_sims: int = 20000, seed: int = 42):
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model = fit_match_model()
    clean = pd.read_pickle(INTERIM / "results_clean.pkl")
    _, ratings = compute_elo(clean)
    ratings = {canonical(k): v for k, v in ratings.items()}
    missing = [t for t in ALL_TEAMS if t not in ratings]
    if missing:
        raise ValueError(f"teams missing from ratings: {missing}")

    sim = GoldenGloveSimulator(model, ratings)
    df = sim.run(n_sims, seed=seed)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS / "golden_glove.csv", index=False)
    (OUTPUTS / "gg_meta.json").write_text(json.dumps(
        {"n_sims": n_sims, "seed": seed, "model": model.to_dict()}, indent=2))

    print(f"ran {n_sims:,} tournaments (seed={seed}) -> outputs/golden_glove.csv\n")
    print("Golden Glove odds (top 20):")
    print(f"{'keeper':22s}{'team':16s}{'glove%':>8s}{'top3%':>8s}{'cleansheet':>12s}{'conceded':>10s}")
    for _, r in df.head(20).iterrows():
        print(f"{r.keeper[:21]:22s}{r.team[:15]:16s}{r.P_golden_glove*100:7.1f}%"
              f"{r.P_top3*100:7.1f}%{r.exp_clean_sheets:12.2f}{r.exp_conceded:10.2f}")
    return df


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(n_sims=n)
