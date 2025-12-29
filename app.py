# app.py
# NCAA Shooting Dashboard (ESPN) — styled like your NBA.com Streamlit app screenshot
#
# What it does
# - Paste an ESPN NCAA men's game play-by-play URL (or gameId)
# - Select Team + Player
# - Shows a clean header with player headshot + team logo + quick shooting line
# - Tabs:
#   1) Zone breakdown (table with FG% color bands like your NBA app)
#   2) Team overview (all players + zone totals)
# - “Refresh data now (pull latest)” button
#
# Install
#   pip install -r requirements.txt
#
# Run
#   streamlit run app.py

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# optional fuzzy name matching
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False


# =========================
# Config
# =========================
USER_AGENT = "Mozilla/5.0"
DEFAULT_GAME_URL = "https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401817514"
CACHE_TTL_SECONDS = 300  # 5 minutes like your NBA app auto-refresh note


# =========================
# ESPN fetch
# =========================
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


# =========================
# Parsing helpers
# =========================
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
            "floater",
            "bank shot",
            "jump shot",
        ]
    )


def shot_points(text: str) -> int:
    """
    Infer points from text.
    - Free throw: 1
    - Three point: 3
    - Otherwise: 2
    """
    t = (text or "").lower()
    if "free throw" in t:
        return 1
    if "three point" in t or "three-point" in t:
        return 3
    return 2


def parse_distance_feet(text: str) -> Optional[int]:
    """
    Many ESPN plays include: 'misses 24-foot three point jumper'
    """
    m = re.search(r"(\d+)-foot", (text or "").lower())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_shooter(text: str) -> str:
    parts = re.split(r"\bmade\b|\bmissed\b|\bmakes\b|\bmisses\b", text, flags=re.IGNORECASE)
    shooter = (parts[0] if parts else "").strip().strip(".")
    return shooter


def infer_zone(text: str) -> str:
    """
    NBA-style zones (best-effort from text only; ESPN PBP often lacks true X/Y coords)
    Zones:
      - Restricted Area
      - In The Paint (Non-RA)
      - Mid-Range
      - Left Corner 3
      - Right Corner 3
      - Above the Break 3
      - Free Throw
      - Other
    """
    t = (text or "").lower()

    if "free throw" in t:
        return "Free Throw"

    is_three = ("three point" in t) or ("three-point" in t)
    dist = parse_distance_feet(text)

    # Corner hints
    if is_three:
        if "left corner" in t:
            return "Left Corner 3"
        if "right corner" in t:
            return "Right Corner 3"
        if "corner" in t:
            return "Corner 3"
        # Many pbp use 22-foot (corner) / 23+ (above break) as rough proxy
        if dist is not None and dist <= 22:
            return "Corner 3"
        return "Above the Break 3"

    # Rim / paint / midrange heuristics
    rim_keywords = ["dunk", "layup", "tip-in", "putback", "alley-oop"]
    if any(k in t for k in rim_keywords):
        # if distance is provided, use it
        if dist is not None:
            if dist <= 3:
                return "Restricted Area"
            if dist <= 10:
                return "In The Paint (Non-RA)"
            if dist <= 22:
                return "Mid-Range"
            return "Other"
        # no distance: treat as restricted/paint
        if "driving" in t or "running" in t:
            return "In The Paint (Non-RA)"
        return "Restricted Area"

    # Jumpers/hook/floater often midrange unless distance says otherwise
    if dist is not None:
        if dist <= 3:
            return "Restricted Area"
        if dist <= 10:
            return "In The Paint (Non-RA)"
        if dist <= 22:
            return "Mid-Range"
        # sometimes ESPN says 24-foot but omits "three point" keyword; treat as 3
        if dist >= 23:
            return "Above the Break 3"
        return "Other"

    if any(k in t for k in ["jumper", "jump shot", "hook shot", "fadeaway", "turnaround", "floater"]):
        return "Mid-Range"

    return "Other"


def parse_shots(pbp_json: Dict[str, Any]) -> pd.DataFrame:
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

        shooter = parse_shooter(text)
        team = None
        if isinstance(p.get("team"), dict):
            team = p["team"].get("displayName") or p["team"].get("abbreviation")

        clock = None
        if isinstance(p.get("clock"), dict):
            clock = p["clock"].get("displayValue")

        period = None
        if isinstance(p.get("period"), dict):
            period = p["period"].get("number")

        zone = infer_zone(text)
        pts = shot_points(text)
        made_flag = 1 if result == "Made" else 0

        rows.append(
            {
                "team": team,
                "period": period,
                "clock": clock,
                "shooter": shooter,
                "result": result,
                "zone": zone,
                "pts": pts * made_flag,     # points scored on that event
                "shot_value": pts,          # 1/2/3
                "description": text,
            }
        )

    return pd.DataFrame(rows)


def get_teams_from_summary(summary_json: Dict[str, Any]) -> List[str]:
    teams = []
    try:
        comps = summary_json.get("header", {}).get("competitions", [])
        if comps:
            competitors = comps[0].get("competitors", [])
            for c in competitors:
                t = (c.get("team") or {}).get("displayName")
                if t:
                    teams.append(t)
    except Exception:
        pass
    return teams


def get_team_logos_from_summary(summary_json: Dict[str, Any]) -> Dict[str, str]:
    """
    Map team displayName -> logo URL (best effort).
    """
    out = {}
    try:
        comps = summary_json.get("header", {}).get("competitions", [])
        if comps:
            competitors = comps[0].get("competitors", [])
            for c in competitors:
                team_obj = c.get("team") or {}
                name = team_obj.get("displayName")
                logos = team_obj.get("logos") or []
                logo = None
                if isinstance(logos, list) and logos:
                    # usually first logo has href
                    if isinstance(logos[0], dict):
                        logo = logos[0].get("href")
                if name and logo:
                    out[name] = logo
    except Exception:
        pass
    return out


def build_headshot_lookup(summary_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Build (team, player_norm) -> headshot_url from ESPN boxscore.
    """
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

    df = pd.DataFrame(rows).drop_duplicates(subset=["team", "player_norm"])
    return df


def attach_headshots(shots: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
    shots["player_norm"] = shots["shooter"].map(normalize_name)

    merged = shots.merge(
        lookup[["team", "player_norm", "headshot_url", "player"]],
        on=["team", "player_norm"],
        how="left",
    )

    # Optional fuzzy match if headshot missing (PBP name slightly different)
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


# =========================
# Aggregations (NBA-style table)
# =========================
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

    df = df.copy()
    df["FGA"] = 1
    df["FGM"] = (df["result"] == "Made").astype(int)

    g = df.groupby("zone", as_index=False).agg(
        FGM=("FGM", "sum"),
        FGA=("FGA", "sum"),
        PTS=("pts", "sum"),
    )
    g["FG%"] = (g["FGM"] / g["FGA"]).fillna(0.0)
    g["PTS/shot"] = (g["PTS"] / g["FGA"]).fillna(0.0)

    total_attempts = float(df["FGA"].sum())
    g["Shot Share"] = (g["FGA"] / total_attempts).fillna(0.0) if total_attempts > 0 else 0.0

    # Pretty formatting
    g.rename(columns={"zone": "Zone"}, inplace=True)

    # Order
    g["__ord"] = g["Zone"].apply(lambda z: ZONE_ORDER.index(z) if z in ZONE_ORDER else 999)
    g = g.sort_values("__ord").drop(columns="__ord")

    return g


def player_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Team overview: each player totals
    """
    if df.empty:
        return pd.DataFrame(columns=["Player", "FGM", "FGA", "FG%", "3PM", "3PA", "3P%", "FTM", "FTA", "FT%", "PTS"])

    df = df.copy()
    df["FGA"] = 1
    df["FGM"] = (df["result"] == "Made").astype(int)

    df["is_3"] = (df["shot_value"] == 3).astype(int)
    df["3PA"] = df["is_3"]
    df["3PM"] = df["is_3"] * df["FGM"]

    df["is_ft"] = (df["shot_value"] == 1).astype(int)
    df["FTA"] = df["is_ft"]
    df["FTM"] = df["is_ft"] * df["FGM"]

    g = df.groupby("shooter", as_index=False).agg(
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
    g = g.sort_values(["PTS", "FGA"], ascending=[False, False])
    return g


def header_line(df_player: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Return (PTS, FG%, 3P%, FT%) for the selected player in this single game.
    """
    if df_player.empty:
        return 0.0, 0.0, 0.0, 0.0

    df = df_player.copy()
    df["FGA"] = 1
    df["FGM"] = (df["result"] == "Made").astype(int)
    pts = float(df["pts"].sum())
    fg = float(df["FGM"].sum() / df["FGA"].sum()) if df["FGA"].sum() else 0.0

    threes = df[df["shot_value"] == 3]
    three_pct = float((threes["result"] == "Made").mean()) if len(threes) else 0.0

    fts = df[df["shot_value"] == 1]
    ft_pct = float((fts["result"] == "Made").mean()) if len(fts) else 0.0

    return pts, fg, three_pct, ft_pct


# =========================
# Styling (FG% bands like screenshot)
# =========================
def fg_band_color(v: float) -> str:
    """
    Red < 0.30, Yellow 0.30-0.40, Green > 0.40
    """
    try:
        x = float(v)
    except Exception:
        return ""
    if x < 0.30:
        return "background-color: #f87171;"   # red-ish
    if x <= 0.40:
        return "background-color: #facc15;"   # yellow-ish
    return "background-color: #4ade80;"       # green-ish


def style_zone_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    show = df.copy()

    # Format columns to match NBA-style feel
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


# =========================
# Streamlit UI (layout like your NBA app)
# =========================
st.set_page_config(page_title="College Basketball Shooting Dashboard", layout="wide")

# Sidebar top refresh button
colA, colB = st.columns([1, 8])
with colA:
    refresh = st.button("Refresh data now (pull latest)")
with colB:
    st.markdown("")

# Title (like your NBA app)
st.title("College Basketball Shooting — ESPN Play-by-Play")

# Sidebar controls
with st.sidebar:
    st.header("Filters")
    game_input = st.text_input("Paste ESPN play-by-play URL or gameId:", value=DEFAULT_GAME_URL)

    # Force refresh: clear cache
    if refresh:
        fetch_game_data.clear()

# Load data
game_id = None
data = None
error = None
try:
    game_id = extract_game_id(game_input)
    data = fetch_game_data(game_id)
except Exception as e:
    error = str(e)

if error:
    st.error(error)
    st.stop()

summary_json = data["summary"]
pbp_json = data["pbp"]

# Build shot table + headshots
shots = parse_shots(pbp_json)
headshots = build_headshot_lookup(summary_json)
shots = attach_headshots(shots, headshots)

team_logos = get_team_logos_from_summary(summary_json)

# If ESPN returns no shots (rare), show a helpful message
if shots.empty:
    st.warning("No shots parsed from ESPN play-by-play for this event.")
    st.stop()

# Build team list from shots (safer than header)
teams = sorted([t for t in shots["team"].dropna().unique().tolist()])

with st.sidebar:
    team_choice = st.selectbox("Choose a team:", options=["All"] + teams, index=0)

# Filter by team
shots_team = shots.copy()
if team_choice != "All":
    shots_team = shots_team[shots_team["team"] == team_choice]

players = sorted([p for p in shots_team["shooter"].dropna().unique().tolist()])

with st.sidebar:
    player_choice = st.selectbox("Choose a player:", options=players, index=0 if players else 0)

# Filter by player
shots_player = shots_team[shots_team["shooter"] == player_choice].copy()

# Header row like your NBA app: photo + logo + name + quick stats line
left, mid, right = st.columns([1.2, 1.2, 6.6])

# Player headshot
with left:
    hs = None
    # pick first non-null
    if "headshot_url" in shots_player.columns:
        vals = shots_player["headshot_url"].dropna().unique().tolist()
        hs = vals[0] if vals else None
    if isinstance(hs, str) and hs.startswith("http"):
        st.image(hs, width=120)

# Team logo (if available)
with mid:
    logo_url = None
    if team_choice != "All":
        logo_url = team_logos.get(team_choice)
    else:
        # just show nothing if All
        logo_url = None
    if isinstance(logo_url, str) and logo_url.startswith("http"):
        st.image(logo_url, width=110)

# Text header
with right:
    # Figure out opponent label from header if available
    st.subheader(f"{player_choice} — Single Game Shooting Profile")
    pts, fg, three_pct, ft_pct = header_line(shots_player)

    # Create a clean line similar to your NBA app
    st.caption(
        f"PTS: {pts:.0f}  |  FG%: {fg*100:.1f}%  |  3P%: {three_pct*100:.1f}%  |  FT%: {ft_pct*100:.1f}%"
    )
    st.caption("FG% color bands: Red < 30%, Yellow 30–40%, Green > 40%")

st.divider()

# Tabs like your NBA app
tab1, tab2 = st.tabs(["Zone breakdown", "Team overview"])

with tab1:
    zb = zone_breakdown(shots_player)

    # add a clean index column like your screenshot
    zb = zb.reset_index(drop=True)
    zb.insert(0, "", zb.index)

    st.dataframe(
        style_zone_table(zb),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    # Team overview table (all players)
    to = player_overview(shots_team).reset_index(drop=True)
    to.insert(0, "", to.index)

    # Make percent columns pretty
    for c in ["FG%", "3P%", "FT%"]:
        if c in to.columns:
            to[c] = (to[c] * 100).round(1)

    st.dataframe(to, use_container_width=True, hide_index=True)

    # Optional: show shot log underneath
    with st.expander("Show shot log (play-by-play shots)"):
        show_cols = ["team", "period", "clock", "shooter", "result", "zone", "shot_value", "pts", "description"]
        st.dataframe(shots_team[show_cols], use_container_width=True, hide_index=True)

# Footer note like your NBA app
st.caption("Auto-updates every 5 min (or press Refresh). ESPN play-by-play is text-based; zone locations are inferred from descriptions.")
