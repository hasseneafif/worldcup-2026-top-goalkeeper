"""2026 World Cup format: official group draw + knockout bracket structure.

Group draw verified against NBC Sports and Olympics.com (draw held 2025-12-05).
Knockout slots and bracket tree from the Wikipedia knockout-stage bracket.

Team names use martj42 canonical spellings (after src.data.clean.canonical()).
"""
from __future__ import annotations

# --- official group draw (canonical names) ---
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# host nations get an effective Elo bump (host advantage ~ +100 in own venues; most
# venues are in the USA, so the bump is largest there). Applied for the whole event.
HOST_ELO_BONUS = {"United States": 60.0, "Mexico": 40.0, "Canada": 40.0}

# --- Round of 32 slots ---
# Each match: (match_no, slot_home, slot_away)
#   ("W", "A")  winner of group A
#   ("R", "A")  runner-up of group A
#   ("3", frozenset({...}))  best third-placed team coming from one of these groups
R32 = [
    (73, ("R", "A"), ("R", "B")),
    (74, ("W", "E"), ("3", frozenset("ABCDF"))),
    (75, ("W", "F"), ("R", "C")),
    (76, ("W", "C"), ("R", "F")),
    (77, ("W", "I"), ("3", frozenset("CDFGH"))),
    (78, ("R", "E"), ("R", "I")),
    (79, ("W", "A"), ("3", frozenset("CEFHI"))),
    (80, ("W", "L"), ("3", frozenset("EHIJK"))),
    (81, ("W", "D"), ("3", frozenset("BEFIJ"))),
    (82, ("W", "G"), ("3", frozenset("AEHIJ"))),
    (83, ("R", "K"), ("R", "L")),
    (84, ("W", "H"), ("R", "J")),
    (85, ("W", "B"), ("3", frozenset("EFGIJ"))),
    (86, ("W", "J"), ("R", "H")),
    (87, ("W", "K"), ("3", frozenset("DEIJL"))),
    (88, ("R", "D"), ("R", "G")),
]

# --- bracket tree: match_no -> (feeder_match_1, feeder_match_2) ---
KO_TREE = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),
}
FINAL = 104
SEMIS = (101, 102)
QUARTERS = (97, 98, 99, 100)
R16 = (89, 90, 91, 92, 93, 94, 95, 96)

ALL_TEAMS = [t for g in GROUPS.values() for t in g]


def group_fixtures(teams: list[str]) -> list[tuple[int, int]]:
    """Round-robin index pairs for a 4-team group (6 matches)."""
    return [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
