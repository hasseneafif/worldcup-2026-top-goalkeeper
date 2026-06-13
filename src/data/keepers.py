"""First-choice goalkeeper per 2026 team.

Goalkeepers can't be derived from goals data, so this is a curated map. The serious
Golden Glove contenders (deep-running teams) are verified against June-2026 squad
news; keepers for weaker teams are best-effort and don't affect the odds materially
(the probability is team-level — the keeper name is just the label on it).

Team names match src.simulate.bracket.ALL_TEAMS (canonical spellings).
"""
from __future__ import annotations

# verified from 2026 squad announcements / pre-tournament news
KEEPERS: dict[str, str] = {
    # contenders (high confidence)
    "Spain": "David Raya",
    "Argentina": "Emiliano Martínez",
    "France": "Mike Maignan",
    "England": "Jordan Pickford",
    "Brazil": "Alisson",
    "Portugal": "Diogo Costa",
    "Belgium": "Thibaut Courtois",
    "Germany": "Manuel Neuer",
    "Netherlands": "Bart Verbruggen",
    "Croatia": "Dominik Livaković",
    "Switzerland": "Gregor Kobel",
    "Morocco": "Yassine Bounou",
    "Japan": "Zion Suzuki",
    "Senegal": "Édouard Mendy",
    "Colombia": "Camilo Vargas",
    "Uruguay": "Sergio Rochet",
    "Mexico": "Luis Malagón",
    "United States": "Matt Turner",
    # mid / best-effort
    "Norway": "Ørjan Nyland",
    "Ecuador": "Hernán Galíndez",
    "Austria": "Alexander Schlager",
    "Australia": "Mathew Ryan",
    "Iran": "Alireza Beiranvand",
    "Egypt": "Mohamed El Shenawy",
    "Turkey": "Uğurcan Çakır",
    "Sweden": "Robin Olsen",
    "Scotland": "Angus Gunn",
    "South Korea": "Jo Hyeon-woo",
    "Canada": "Dayne St. Clair",
    "Czech Republic": "Jindřich Staněk",
    "Ivory Coast": "Yahia Fofana",
    "Algeria": "Anthony Mandrea",
    "Paraguay": "Roberto Fernández",
    "Saudi Arabia": "Mohammed Al-Owais",
    "Tunisia": "Aymen Dahmen",
    "Ghana": "Lawrence Ati-Zigi",
    "South Africa": "Ronwen Williams",
    "Qatar": "Meshaal Barsham",
    "Bosnia and Herzegovina": "Nikola Vasilj",
    "DR Congo": "Lionel Mpasi",
    "Panama": "Orlando Mosquera",
    "Iraq": "Jalal Hassan",
    "Jordan": "Yazeed Abulaila",
    "Uzbekistan": "Utkir Yusupov",
    "Cape Verde": "Vozinha",
    "Curaçao": "Eloy Room",
    "New Zealand": "Max Crocombe",
    "Haiti": "Johny Placide",
}


def keeper_for(team: str) -> str:
    return KEEPERS.get(team, f"({team} GK)")
