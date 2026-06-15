# 2026 World Cup — Golden Glove Predictor

![Golden Glove](https://i.imgur.com/Rehvln6.png)

Estimates each goalkeeper's probability of winning the **Golden Glove** (best keeper)
at the 2026 FIFA World Cup. The award is driven by clean sheets, so the model is:

> **team defensive strength × clean sheets × tournament depth**

A keeper on a strong, deep-running defence gets the most chances to keep clean sheets —
which is why the Golden Glove almost always goes to a semi-finalist or finalist's
number one (Martínez 2022, Lloris 2018, Courtois 2018 runner-up, Neuer 2014).

## How it works

1. **Match goals** — the reused Dixon-Coles + Elo match model gives each team's expected
   goals conceded in every simulated match (a function of the opponent's strength).
2. **Clean sheets** (`src/simulate/goldenglove.py`) — over 20,000 simulated tournaments,
   each team's goals conceded and clean sheets are tracked across every match it plays
   (a 0-0 won on penalties still counts as a clean sheet for both keepers).
3. **Winner** — per tournament, the keeper with the most clean sheets wins (ties broken
   by fewest goals conceded). Aggregating gives each team's Golden Glove probability.
4. **Keeper label** — odds are computed at team level; each team's first-choice keeper
   (`src/data/keepers.py`) is attached as the name.

## Quickstart

```bash
# uses the same deps as the sibling models (pandas/numpy/scipy)
python -m src.data.clean              # -> data/interim/results_clean.pkl
python -m src.models.elo              # -> data/interim/results_elo.pkl
python -m src.simulate.goldenglove 20000   # -> outputs/golden_glove.csv
```

## Results (20,000 simulations)

| # | Keeper | Team | Golden Glove | Top-3 | Exp. clean sheets |
|---|--------|------|:------------:|:-----:|:-----------------:|
| 1 | Unai Simón | Spain | **19.2%** | 39.3% | 3.41 |
| 2 | Emiliano Martínez | Argentina | **13.1%** | 29.4% | 3.01 |
| 3 | Mike Maignan | France | **8.3%** | 21.4% | 2.66 |
| 4 | Jordan Pickford | England | **6.5%** | 17.5% | 2.48 |
| 5 | Alisson | Brazil | **4.8%** | 14.4% | 2.30 |
| 6 | Camilo Vargas | Colombia | **4.5%** | 13.0% | 2.18 |
| 7 | Luis Malagón | Mexico | **4.4%** | 14.2% | 2.32 |
| 8 | Diogo Costa | Portugal | **3.8%** | 11.8% | 2.09 |
| 9 | Hernán Galíndez | Ecuador | **3.7%** | 12.4% | 2.18 |
| 10 | Manuel Neuer | Germany | **2.8%** | 9.9% | 2.05 |

Full table in `outputs/golden_glove.csv`. Reigning Golden Glove winner Emiliano Martínez
ranks 2nd behind Unai Simón, whose Spain side is the model's strongest and deepest-running team.

## Data

- `results.csv` — martj42 (free, GitHub). No Kaggle token required.
- The match engine (`clean.py`, `elo.py`, `features.py`, `poisson.py`, `bracket.py`) is
  shared with the sibling title-odds and Golden Boot models.
- `keepers.py` — curated first-choice keepers; contenders verified against June-2026
  squad news, weaker teams best-effort (immaterial to the odds).

## Known limitations / next steps

- No **save / shot-stopping** data in the free source, so clean sheets (not save %) drive
  the model. The real award also weighs saves and an expert panel.
- Assumes the first-choice keeper plays every match (no rotation or in-tournament injury).
- A **news/LLM layer** (as in the title model) would handle keeper injuries and late
  changes — the main thing that would shift these odds.
