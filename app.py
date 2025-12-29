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
CACHE_TTL_SECONDS = 300

DEFAULT_GAME_URL = "https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401817514"
DEFAULT_ROSTER_TEAM_ID = "120"  # Maryland
DEFAULT_ROSTER_TEAM_NAME = "Maryland Terrapins"


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


def safe_get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r


def safe_get_json(url: str) -> Dict[str, Any]:
    r = safe_get(url)
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
    return safe_get_json(
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/roster"
    )


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_pbp_html_tables(game_id: str) -> List[pd.DataFrame]:
    """
    Fallback: scrape ESPN play-by-play webpage tables when JSON API returns 0 plays.
    """
    url = f"https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/{game_id}"
    html = safe_get(url).text
    # ESPN has multiple tables (often one per half/OT)
    tables = pd.read_html(html)
    return tables


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


def parse_shots_from_api(pbp_json: Dict[str, Any]) -> pd.DataFrame:
    schema = ["team", "period", "clock", "shooter", "result", "zone", "shot_value", "pts", "description"]
    plays = extract_all_plays(pbp_json)

    rows = []
    for p in plays:
        if not isinstance(p, dict):
            continue

        text = (p.get("text") or "").strip()
        if not text or not is_shot(text):
            continue

        result = parse_result(text)
        if not result:
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
            dict(
                team=team, period=period, clock=clock, shooter=shooter, result=result,
                zone=zone, shot_value=val, pts=pts, description=text
            )
        )

    df = pd.DataFrame(rows)
    for c in schema:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df[schema]


def teams_from_summary(summary_json: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (away_name, home_name)
    """
    away = ""
    home = ""
    comps = summary_json.get("header", {}).get("competitions", [])
    if comps:
        competitors = comps[0].get("competitors", []) or []
        # usually: 0 away, 1 home but we won't assume order
        for c in competitors:
            team = c.get("team") or {}
            name = team.get("displayName") or ""
            ha = (c.get("homeAway") or "").lower()
            if ha == "away":
                away = name
            elif ha == "home":
                home = name
    return away, home


def parse_shots_from_html_tables(summary_json: Dict[str, Any], tables: List[pd.DataFrame]) -> pd.DataFrame:
    """
    ESPN HTML pbp tables typically have:
      - a time column
      - one column for away text
      - one column for home text
    We'll detect those dynamically.
    """
    schema = ["team", "period", "clock", "shooter", "result", "zone", "shot_value", "pts", "description"]

    away_name, home_name = teams_from_summary(summary_json)

    rows = []
    period_guess = 1

    for t in tables:
        if t is None or t.empty:
            continue

        # Normalize column names
        cols = [str(c).strip().lower() for c in t.columns]
        df = t.copy()
        df.columns = cols

        # Find a likely clock column
        clock_col = None
        for c in cols:
            if "time" in c or "clock" in c:
                clock_col = c
                break
        if clock_col is None:
            # sometimes the first column is time
            clock_col = cols[0]

        # Find away/home text columns:
        # (usually two text columns besides the clock)
        text_cols = [c for c in cols if c != clock_col]
        if len(text_cols) < 2:
            # not a pbp table
            continue

        away_col = text_cols[0]
        home_col = text_cols[1]

        for _, r in df.iterrows():
            clock = r.get(clock_col, None)

            away_txt = r.get(away_col, None)
            home_txt = r.get(home_col, None)

            # pick whichever side has text
            team = None
            text = None

            if isinstance(away_txt, str) and away_txt.strip():
                team = away_name or "Away"
                text = away_txt.strip()
            elif isinstance(home_txt, str) and home_txt.strip():
                team = home_name or "Home"
                text = home_txt.strip()

            if not text:
                continue

            if not is_shot(text):
                continue

            result = parse_result(text)
            if not result:
                continue

            shooter = parse_shooter(text)
            val = shot_value(text)
            zone = infer_zone(text)
            pts = val if result == "Made" else 0

            rows.append(
                dict(
                    team=team,
                    period=period_guess,  # HTML tables don't always label the period cleanly; this is a usable proxy
                    clock=clock,
                    shooter=shooter,
                    result=result,
                    zone=zone,
                    shot_value=val,
                    pts=pts,
                    description=text,
                )
            )

        period_guess += 1

    df_out = pd.DataFrame(rows)
    for c in schema:
        if c not in df_out.columns:
            df_out[c] = pd.Series(dtype="object")
    return df_out[schema]


# -----------------------------
# Roster headshots lookup
# -----------------------------
def build_roster_headshots(team_id: str, team_name: str) -> pd.DataFrame:
    roster_json = fetch_team_roster(team_id)
    rows = []
    for a in roster_json.get("athletes", []) or []:
        name = a.get("displayName") or a.get("fullName")
        aid = a.get("id")
        if name and aid:
            rows.append(
                {
                    "player": name,
                    "player_norm": normalize_name(name),
                    "headshot_url": f"https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/{aid}.png",
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["player_norm"])
    return df


def attach_headshots(shots: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
    shots["player_norm"] = shots["shooter"].map(normalize_name)

    merged = shots.merge(
        lookup[["player_norm", "headshot_url", "player"]],
        on="player_norm",
        how="left",
    )

    if RAPIDFUZZ_AVAILABLE and not lookup.empty:
        missing = merged["headshot_url"].isna()
        if missing.any():
            fixed = merged.copy()
            choices = list(lookup["player_norm"].unique())
            for i in fixed.index[missing].tolist():
                target = fixed.at[i, "player_norm"]
                if not target:
                    continue
                match = process.extractOne(target, choices, scorer=fuzz.WRatio)
                if match and match[1] >= 88:
                    best = match[0]
                    row = lookup[lookup["player_norm"] == best].head(1)
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
        show["FG%"] = pd.to_numeric(show["FG%"], errors="coerce").fillna(0.0).round(3)
    if "PTS/shot" in show.columns:
        show["PTS/shot"] = pd.to_numeric(show["PTS/shot"], errors="coerce").fillna(0.0).round(2)
    if "Shot Share" in show.columns:
        ss = pd.to_numeric(show["Shot Share"], errors="coerce").fillna(0.0)
        show["Shot Share"] = (ss * 100).round(0).astype(int).astype(str) + "%"

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

top_left, top_right = st.columns([1.6, 8.4])
with top_left:
    refresh = st.button("Refresh data now\n(pull latest)")
with top_right:
    st.markdown("")

st.title("College Basketball Shooting — ESPN Play-by-Play")

with st.sidebar:
    st.header("Filters")
    game_input = st.text_input("Paste ESPN play-by-play URL or gameId:", value=DEFAULT_GAME_URL)

    st.subheader("Roster (headshots fallback)")
    roster_team_id = st.text_input("Roster teamId (ex: Maryland = 120)", value=DEFAULT_ROSTER_TEAM_ID)
    roster_team_name = st.text_input("Roster team name", value=DEFAULT_ROSTER_TEAM_NAME)

    if refresh:
        fetch_game_data.clear()
        fetch_team_roster.clear()
        fetch_pbp_html_tables.clear()

game_id = extract_game_id(game_input)

data = fetch_game_data(game_id)
summary_json = data["summary"]
pbp_json = data["pbp"]

plays_found = len(extract_all_plays(pbp_json))

# parse from API first
shots_api = parse_shots_from_api(pbp_json)

# if API is empty, scrape HTML tables
shots = shots_api
source_used = "ESPN JSON API"
if plays_found == 0 or len(shots_api) == 0:
    try:
        tables = fetch_pbp_html_tables(game_id)
        shots_html = parse_shots_from_html_tables(summary_json, tables)
        if len(shots_html) > 0:
            shots = shots_html
            source_used = "ESPN HTML (pandas.read_html fallback)"
    except Exception:
        pass

# roster lookup always available
roster_lookup = build_roster_headshots(roster_team_id.strip(), roster_team_name.strip())

st.caption(
    f"Game ID: {game_id} | API plays found: {plays_found} | Shots parsed: {len(shots)} | Source: {source_used}"
)

# attach headshots if shots exist
if len(shots) > 0:
    shots = attach_headshots(shots, roster_lookup)

# if STILL no shots, show roster grid and stop
if len(shots) == 0:
    st.warning(
        "Could not parse shots from ESPN API or HTML tables for this game. "
        "Showing roster headshots so the app still looks good."
    )

    st.subheader(f"{roster_team_name} — Roster Headshots")
    roster_show = roster_lookup.sort_values("player")

    cols = st.columns(6)
    for i, (_, r) in enumerate(roster_show.iterrows()):
        with cols[i % 6]:
            st.image(r["headshot_url"], width=95)
            st.caption(r["player"])
    st.stop()

teams = sorted([t for t in shots["team"].dropna().unique().tolist()])
with st.sidebar:
    team_choice = st.selectbox("Choose a team:", options=["All"] + teams, index=0)

shots_team = shots.copy()
if team_choice != "All":
    shots_team = shots_team[shots_team["team"] == team_choice]

players = sorted([p for p in shots_team["shooter"].dropna().unique().tolist()])
with st.sidebar:
    player_choice = st.selectbox("Choose a player:", options=players, index=0 if players else 0)

shots_player = shots_team[shots_team["shooter"] == player_choice].copy()

# Header
left, right = st.columns([1.5, 8.5])
with left:
    vals = shots_player["headshot_url"].dropna().unique().tolist() if "headshot_url" in shots_player.columns else []
    hs = vals[0] if vals else None
    if isinstance(hs, str) and hs.startswith("http"):
        st.image(hs, width=120)

with right:
    st.subheader(f"{player_choice} — Single Game Shooting Profile")
    pts, fg, three_pct, ft_pct = header_line(shots_player)
    st.caption(f"PTS: {pts}  |  FG%: {fg*100:.1f}%  |  3P%: {three_pct*100:.1f}%  |  FT%: {ft_pct*100:.1f}%")
    st.caption("FG% color bands: Red < 30%, Yellow 30–40%, Green > 40%")

st.divider()

tab1, tab2 = st.tabs(["Zone breakdown", "Shot log"])

with tab1:
    zb = zone_breakdown(shots_player).reset_index(drop=True)
    zb.insert(0, "", zb.index)
    st.dataframe(style_zone_table(zb), use_container_width=True, hide_index=True)

with tab2:
    show_cols = ["team", "period", "clock", "shooter", "result", "zone", "shot_value", "pts", "description"]
    if "headshot_url" in shots_team.columns:
        show_cols.append("headshot_url")
    st.dataframe(shots_team[show_cols], use_container_width=True, hide_index=True)

st.caption("Auto-updates every 5 min (or press Refresh). Zones are inferred from descriptions.")
