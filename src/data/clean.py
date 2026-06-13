"""Load and normalize the raw match results.

The martj42 results.csv already uses clean modern country names, so the canonical
map here mostly handles *cross-source* spellings (Elo / Transfermarkt / odds feeds)
and a couple of encoding quirks. Route every team name from any source through
`canonical()` so joins never silently miss.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"

# alternate spelling -> canonical name (canonical = how martj42 results.csv spells it)
ALIASES = {
    "USA": "United States",
    "US": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "IR Iran": "Iran",
    "Iran (Islamic Republic of)": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Cura�ao": "Curaçao",  # mojibake from a bad decode upstream
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "DRC": "DR Congo",
    "Republic of Ireland": "Republic of Ireland",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cape Verde Islands": "Cape Verde",
    "North Macedonia": "North Macedonia",
    "FYR Macedonia": "North Macedonia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "United States of America": "United States",
}


def canonical(name: str) -> str:
    """Normalize a single team name to the canonical spelling."""
    if name is None:
        return name
    n = str(name).strip()
    return ALIASES.get(n, n)


def load_results(path: Path | None = None) -> pd.DataFrame:
    """Read results.csv, fix encoding/dates, canonicalize names, add helper columns."""
    path = path or (RAW / "results.csv")
    df = pd.read_csv(path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()

    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])

    # de-duplicate exact repeats, sort chronologically (Elo needs time order)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["result"] = df.apply(
        lambda r: "H" if r.home_score > r.away_score else ("A" if r.home_score < r.away_score else "D"),
        axis=1,
    )
    return df


def verify_teams(df: pd.DataFrame, teams: list[str]) -> dict[str, bool]:
    """Return {team: appears_in_data}. Raise if any are missing."""
    known = set(df["home_team"]) | set(df["away_team"])
    status = {t: (canonical(t) in known) for t in teams}
    missing = [t for t, ok in status.items() if not ok]
    if missing:
        raise ValueError(f"{len(missing)} team(s) not found in results data: {missing}")
    return status


def main() -> None:
    df = load_results()
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / "results_clean.pkl"
    df.to_pickle(out)
    print(f"cleaned {len(df):,} matches -> {out}")
    print(f"date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"unique teams: {pd.concat([df['home_team'], df['away_team']]).nunique()}")


if __name__ == "__main__":
    main()
