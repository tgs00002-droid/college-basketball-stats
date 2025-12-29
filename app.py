import re
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
USER_AGENT = "Mozilla/5.0"
CACHE_TTL_SECONDS = 300  # 5 minutes
DEFAULT_GAME_URL = "https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401817514"


# -----------------------------
# ESPN fetch helpers
# -----------------------------
def extract_game_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if s.isdigit():
        return s
    m = re.search(r"gameId/(\d+)", s)
    if m:
        return m.group(1)
    raise ValueError("Could not find gameId. Paste an ESPN URL with /gameId/######### or just the numeric id.")


def safe_get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_game_data(game_id: str) -> Dict[str, Any]:
    summary = safe_get_json(
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}"
    )
    pbp = safe_get_json(
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/playbyplay?event={game_id}"
    )
    return {"summary": summary, "pbp": pbp}


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_team_roster(team_id: str) -> Dict[str, Any]:
    # This is the JSON equivalent of the roster page you linked
    return safe_get_json(
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/roster"
    )


# -----------------------------
# Parsing helpers
# -----------------------------
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^a-z\s\-']", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def extract_all_plays(pbp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    plays: List[Dict[str, Any]] = []

    if isinstance(pbp_json.get("plays"), list):
        plays.extend(pbp_json["plays"])

    drives = pbp_json.get("drives", [])
    if isinstance(drives, list):
        for d in drives:
            if isinstance(d, dict) and isinstance(d.get("plays"), list):
                plays.extend(d["plays"])

    periods = pbp_json.get("periods", [])
    if isinstance(periods, list):
        for prd in periods:
            if isinstance(prd, dict) and isinstance(prd.get("plays"), list):
                plays.extend(prd["plays"])

    return plays


def parse_result(text: str) -> str:
    t = f" {text.lower()} "
    if " made " in t or " makes " in t:
        return "Made"
    if " missed " in t or " misses " in t:
        return "Missed"
    return ""


def is_shot(text: str) -> bool:
    t = (text or "").lower()
    if not any(v in t for v in ["made", "missed", "makes", "misses"]):
        return False
    return any(
        k in t
        for k in [
            "jumper",
            "three point",
            "three-point",
            "layup",
            "dunk",
            "free throw",
            "tip-in",
            "hook shot",
            "putback",
            "alley-oop",
            "jump shot",
        ]
    )


def parse_shooter(text: str) -> str:
    parts = re.split(r"\bmade\b|\bmissed\b|\bmakes\b|\bmisses\b", text, flags=re.IGNORECASE)
    return (parts[0] if parts else "").strip().strip(".")


def parse_distance_feet(text: str) -> Optional[int]:
    m = re.search(r"(\d+)-foot", (text or "").lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def shot_value(text: str) -> int:
    t = (text or "").lower()
    if "free throw" in t:
        return 1
    if "three point" in t or "three-point" in t:
        return 3
    # sometimes ESPN uses distance but forgets "three point"
    dist = parse_distance_feet(text)
    if dist is not None and dist >= 23:
        return 3
    return 2


def infer_zone(text: str) -> str:
    t = (text or "").lower()

    if "free throw" in t:
        return "Free Throw"

    is_three = ("three point" in t) or ("three-point" in t)
    dist = parse_distance_feet(text)

    if is_three:
        if "left corner" in t:
            return "Left Corner 3"
        if "right corner" in t:
            return "Right Corner 3"
        if "corner" in t:
            return "Corner 3"
        if dist is not None and dist <= 22:
            return "Corner 3"
        return "Above the Break 3"

    rim_words = ["dunk", "layup", "tip-in", "putback", "alley-oop"]
    if any(k in t for k in rim_words):
        if dist is not None:
            if dist <= 3:
                return "Restricted Area"
            if dist <= 10:
                return "In The Paint (Non-RA)"
            if dist <= 22:
                return "Mid-Range"
            return "Other"
        if "driving" in t:
            return "In The Paint (Non-RA)"
        return "Restricted Area"

    if dist is not None:
        if dist <= 3:
            return "Restricted Area"
        if dist <= 10:
            return "In The Paint (Non-RA)"
        if dist <= 22:
            return "Mid-Range"
        if dist >= 23:
            return "Above the Break 3"
        return "Other"

    if any(k in t for k in ["jumper", "hook shot", "fadeaway", "turnaround", "jump shot"]):
        return "Mid-Range"

    return "Other"


def parse_shots(pbp_json: Dict[str, Any]) -> pd.DataFrame:
    """
    IMPORTANT: Always return a DF with the full schema, even when empty.
    This prevents KeyError: 'shooter' downstream.
    """
    schema = [
        "team", "period", "clock", "shooter", "result",
        "zone", "shot_value", "pts", "description"
    ]

    plays = extract_all_plays(pbp_json)
    rows = []

    for p in plays:
        if not isinstance(p, dict):
            continue

        text = (p.get("text") or "").strip()
        if not text or not is_shot(text):
            continue

        result = parse_result(text)
        if result == "":
            continue

        team = None
        if isinstance(p.get("team"), dict):
            team = p["team"].get("displayName") or p["team"].get("abbreviation")

        clock = None
        if isinstance(p.get("clock"), dict):
            clock = p["clock"].get("displayValue")

        period = None
        if isinstance(p.get("period"), dict):
            period = p["period"].get("number")

        shooter = parse_shooter(text)
        val = shot_value(text)
        zone = infer_zone(text)
        pts = val if result == "Made" else 0

        rows.append(
            {
                "team": team,
                "period": period,
                "clock": clock,
                "shooter": shooter,
                "result": result,
                "zone": zone,
                "shot_value": val,
                "pts": pts,
                "description": text,
            }
        )

    df = pd.DataFrame(rows)
    # enforce schema even if empty
    for c in schema:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df[schema]


# -----------------------------
# Headshots: Boxscore + Roster
# -----------------------------
def team_ids_and_logos(summary_json: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      team_name -> team_id
      team_name -> logo_url
    """
    name_to_id = {}
    name_to_logo = {}
    comps = summary_json.get("header", {}).get("competitions", [])
    if not comps:
        return name_to_id, name_to_logo

    competitors = comps[0].get("competitors", []) or []
    for c in competitors:
        team = c.get("team") or {}
        name = team.get("displayName")
        tid = team.get("id")
        logos = team.get("logos") or []
        logo = None
        if isinstance(logos, list) and logos and isinstance(logos[0], dict):
            logo = logos[0].get("href")

        if name and tid:
            name_to_id[name] = str(tid)
        if name and logo:
            name_to_logo[name] = logo

    return name_to_id, name_to_logo


def build_headshot_lookup_from_boxscore(summary_json: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    players_blocks = summary_json.get("boxscore", {}).get("players", [])
    if not isinstance(players_blocks, list):
        return pd.DataFrame(columns=["team", "player", "player_norm", "headshot_url"])

    for team_block in players_blocks:
        team_name = (team_block.get("team") or {}).get("displayName") or (team_block.get("team") or {}).get("abbreviation")
        for stat_group in team_block.get("statistics", []) or []:
            for athlete_row in stat_group.get("athletes", []) or []:
                athlete = (athlete_row or {}).get("athlete") or {}
                name = athlete.get("displayName") or athlete.get("shortName")
                aid = athlete.get("id")
                if name and aid:
                    rows.append(
                        {
                            "team": team_name,
                            "player": name,
                            "player_norm": normalize_name(name),
                            "headshot_url": f"https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/{aid}.png",
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates(subset=["team", "player_norm"])


def build_headshot_lookup_from_roster(team_name: str, roster_json: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    athletes = roster_json.get("athletes", []) or []
    for a in athletes:
        name = a.get("displayName") or a.get("fullName")
        aid = a.get("id")
        if name and aid:
            rows.append(
                {
                    "team": team_name,
                    "player": name,
                    "player_norm": normalize_name(name),
                    "headshot_url": f"https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/{aid}.png",
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["team", "player_norm"])


def build_combined_headshot_lookup(summary_json: Dict[str, Any]) -> pd.DataFrame:
    # Boxscore
    box_df = build_headshot_lookup_from_boxscore(summary_json)

    # Roster fallback for both teams in the game
    team_id_map, _ = team_ids_and_logos(summary_json)
    roster_frames = []
    for team_name, tid in team_id_map.items():
        try:
            roster_json = fetch_team_roster(tid)
            roster_frames.append(build_headshot_lookup_from_roster(team_name, roster_json))
        except Exception:
            pass

    roster_df = pd.concat(roster_frames, ignore_index=True) if roster_frames else pd.DataFrame(
        columns=["team", "player", "player_norm", "headshot_url"]
    )

    combined = pd.concat([box_df, roster_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["team", "player_norm"])
    return combined


def attach_headshots(shots: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
    if "shooter" not in shots.columns:
        # ultra-safety
        shots["shooter"] = ""
    shots["player_norm"] = shots["shooter"].map(normalize_name)

    merged = shots.merge(
        lookup[["team", "player_norm", "headshot_url", "player"]],
        on=["team", "player_norm"],
        how="left",
    )

    # Optional fuzzy matching if still missing
    if RAPIDFUZZ_AVAILABLE and not lookup.empty:
        missing = merged["headshot_url"].isna()
        if missing.any():
            fixed = merged.copy()
            for team in fixed.loc[missing, "team"].dropna().unique():
                team_lookup = lookup[lookup["team"] == team]
                choices = list(team_lookup["player_norm"].unique())
                if not choices:
                    continue
                idxs = fixed.index[(fixed["team"] == team) & (fixed["headshot_url"].isna())].tolist()
                for i in idxs:
                    target = fixed.at[i, "player_norm"]
                    if not target:
                        continue
                    match = process.extractOne(target, choices, scorer=fuzz.WRatio)
                    if match and match[1] >= 88:
                        best = match[0]
                        row = team_lookup[team_lookup["player_norm"] == best].head(1)
                        if not row.empty:
                            fixed.at[i, "headshot_url"] = row["headshot_url"].iloc[0]
                            fixed.at[i, "player"] = row["player"].iloc[0]
            merged = fixed

    return merged


# -----------------------------
# Tables & styling
# -----------------------------
ZONE_ORDER = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Corner 3",
    "Above the Break 3",
    "Free Throw",
    "Other",
]


def zone_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Zone", "FGM", "FGA", "PTS/shot", "FG%", "PTS", "Shot Share"])

    x = df.copy()
    x["FGA"] = 1
    x["FGM"] = (x["result"] == "Made").astype(int)

    g = x.groupby("zone", as_index=False).agg(
        FGM=("FGM", "sum"),
        FGA=("FGA", "sum"),
        PTS=("pts", "sum"),
    )
    g["FG%"] = (g["FGM"] / g["FGA"]).fillna(0.0)
    g["PTS/shot"] = (g["PTS"] / g["FGA"]).fillna(0.0)

    total = float(x["FGA"].sum())
    g["Shot Share"] = (g["FGA"] / total).fillna(0.0) if total > 0 else 0.0

    g.rename(columns={"zone": "Zone"}, inplace=True)
    g["__ord"] = g["Zone"].apply(lambda z: ZONE_ORDER.index(z) if z in ZONE_ORDER else 999)
    g = g.sort_values("__ord").drop(columns="__ord")
    return g


def player_overview(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Player", "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%", "PTS"])

    x = df.copy()
    x["FGA"] = 1
    x["FGM"] = (x["result"] == "Made").astype(int)

    x["3PA"] = (x["shot_value"] == 3).astype(int)
    x["3PM"] = x["3PA"] * x["FGM"]
    x["FTA"] = (x["shot_value"] == 1).astype(int)
    x["FTM"] = x["FTA"] * x["FGM"]

    g = x.groupby("shooter", as_index=False).agg(
        FGM=("FGM", "sum"),
        FGA=("FGA", "sum"),
        PTS=("pts", "sum"),
        **{"3PM": ("3PM", "sum"), "3PA": ("3PA", "sum")},
        FTM=("FTM", "sum"),
        FTA=("FTA", "sum"),
    )

    g["FG%"] = (g["FGM"] / g["FGA"]).fillna(0.0)
    g["3P%"] = (g["3PM"] / g["3PA"]).replace([pd.NA, float("inf")], 0.0).fillna(0.0)
    g["FT%"] = (g["FTM"] / g["FTA"]).replace([pd.NA, float("inf")], 0.0).fillna(0.0)

    g.rename(columns={"shooter": "Player"}, inplace=True)
    return g.sort_values(["PTS", "FGA"], ascending=[False, False])


def fg_band_color(v: float) -> str:
    try:
        x = float(v)
    except Exception:
        return ""
    if x < 0.30:
        return "background-color: #f87171;"
    if x <= 0.40:
        return "background-color: #facc15;"
    return "background-color: #4ade80;"


def style_zone_table(df: pd.DataFrame):
    show = df.copy()
    if "FG%" in show.columns:
        show["FG%"] = show["FG%"].round(3)
    if "PTS/shot" in show.columns:
        show["PTS/shot"] = show["PTS/shot"].round(2)
    if "Shot Share" in show.columns:
        show["Shot Share"] = (show["Shot Share"] * 100).round(0).astype(int).astype(str) + "%"

    styler = show.style
    if "FG%" in show.columns:
        styler = styler.applymap(fg_band_color, subset=["FG%"])
    return styler


def header_line(df_player: pd.DataFrame) -> Tuple[int, float, float, float]:
    if df_player.empty:
        return 0, 0.0, 0.0, 0.0

    pts = int(df_player["pts"].sum())
    fga = len(df_player)
    fgm = int((df_player["result"] == "Made").sum())
    fg = (fgm / fga) if fga else 0.0

    threes = df_player[df_player["shot_value"] == 3]
    three_pct = float((threes["result"] == "Made").mean()) if len(threes) else 0.0

    fts = df_player[df_player["shot_value"] == 1]
    ft_pct = float((fts["result"] == "Made").mean()) if len(fts) else 0.0

    return pts, fg, three_pct, ft_pct


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="College Basketball Shooting — ESPN", layout="wide")

top_left, top_right = st.columns([1.5, 8.5])
with top_left:
    refresh = st.button("Refresh data now\n(pull latest)")
with top_right:
    st.markdown("")

st.title("College Basketball Shooting — ESPN Play-by-Play")

with st.sidebar:
    st.header("Filters")
    game_input = st.text_input("Paste ESPN play-by-play URL or gameId:", value=DEFAULT_GAME_URL)

    if refresh:
        fetch_game_data.clear()
        fetch_team_roster.clear()

# Load game
game_id = extract_game_id(game_input)
data = fetch_game_data(game_id)

summary_json = data["summary"]
pbp_json = data["pbp"]

# Parse
shots = parse_shots(pbp_json)

plays_found = len(extract_all_plays(pbp_json))
st.caption(f"Game ID: {game_id} | Plays found: {plays_found} | Shots parsed: {len(shots)}")

# Headshots (boxscore + roster)
headshots = build_combined_headshot_lookup(summary_json)
shots = attach_headshots(shots, headshots)

# Teams list from shots
teams = sorted([t for t in shots["team"].dropna().unique().tolist()])
team_id_map, team_logo_map = team_ids_and_logos(summary_json)

with st.sidebar:
    team_choice = st.selectbox("Choose a team:", options=["All"] + teams, index=0)

shots_team = shots.copy()
if team_choice != "All":
    shots_team = shots_team[shots_team["team"] == team_choice]

players = sorted([p for p in shots_team["shooter"].dropna().unique().tolist()])
with st.sidebar:
    player_choice = st.selectbox("Choose a player:", options=players, index=0 if players else 0)

shots_player = shots_team[shots_team["shooter"] == player_choice].copy()

# Header (player headshot + team logo + stats line)
left, mid, right = st.columns([1.2, 1.2, 6.6])

with left:
    hs = None
    vals = shots_player["headshot_url"].dropna().unique().tolist() if not shots_player.empty else []
    hs = vals[0] if vals else None
    if isinstance(hs, str) and hs.startswith("http"):
        st.image(hs, width=120)

with mid:
    logo_url = team_logo_map.get(team_choice) if team_choice != "All" else None
    if isinstance(logo_url, str) and logo_url.startswith("http"):
        st.image(logo_url, width=110)

with right:
    st.subheader(f"{player_choice} — Single Game Shooting Profile")
    pts, fg, three_pct, ft_pct = header_line(shots_player)
    st.caption(f"PTS: {pts}  |  FG%: {fg*100:.1f}%  |  3P%: {three_pct*100:.1f}%  |  FT%: {ft_pct*100:.1f}%")
    st.caption("FG% color bands: Red < 30%, Yellow 30–40%, Green > 40%")

st.divider()

tab1, tab2 = st.tabs(["Zone breakdown", "Team overview"])

with tab1:
    zb = zone_breakdown(shots_player).reset_index(drop=True)
    zb.insert(0, "", zb.index)
    st.dataframe(style_zone_table(zb), use_container_width=True, hide_index=True)

with tab2:
    to = player_overview(shots_team).reset_index(drop=True)
    to.insert(0, "", to.index)

    for c in ["FG%", "3P%", "FT%"]:
        if c in to.columns:
            to[c] = (to[c] * 100).round(1)

    st.dataframe(to, use_container_width=True, hide_index=True)

    with st.expander("Show shot log (play-by-play shots)"):
        show_cols = ["team", "period", "clock", "shooter", "result", "zone", "shot_value", "pts", "description", "headshot_url"]
        st.dataframe(shots_team[show_cols], use_container_width=True, hide_index=True)

st.caption("Auto-updates every 5 min (or press Refresh). ESPN play-by-play is text-based; zones are inferred from descriptions.")
