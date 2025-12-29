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


USER_AGENT = "Mozilla/5.0"
CACHE_TTL_SECONDS = 300

DEFAULT_GAME_URL = "https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401817514"
MARYLAND_TEAM_ID = "120"
MARYLAND_NAME_DEFAULT = "Maryland Terrapins"


def safe_get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r


def safe_get_json(url: str) -> Dict[str, Any]:
    return safe_get(url).json()


def extract_game_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if s.isdigit():
        return s
    m = re.search(r"gameId/(\d+)", s)
    if m:
        return m.group(1)
    m2 = re.search(r"gameId=(\d+)", s)
    if m2:
        return m2.group(1)
    raise ValueError("Could not find gameId. Paste an ESPN URL with /gameId/######### or just the numeric id.")


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
    url = f"https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/{game_id}"
    html = safe_get(url).text
    return pd.read_html(html)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_team_schedule(team_id: str) -> Dict[str, Any]:
    return safe_get_json(
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/schedule"
    )


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^a-z\s\-']", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def keep_only_team(shots_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    if shots_df is None or shots_df.empty:
        return shots_df
    tn = (team_name or "").strip().lower()
    if not tn:
        return shots_df

    s = shots_df.copy()
    s["team_norm"] = s["team"].astype(str).str.lower().str.strip()

    mask = s["team_norm"].str.contains(re.escape(tn), na=False)

    if not mask.any() and "maryland" in tn:
        mask = s["team_norm"].str.contains("maryland", na=False)

    s = s.loc[mask].drop(columns=["team_norm"])
    return s


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


def teams_from_summary(summary_json: Dict[str, Any]) -> Tuple[str, str]:
    away = ""
    home = ""
    comps = summary_json.get("header", {}).get("competitions", [])
    if comps:
        competitors = comps[0].get("competitors", []) or []
        for c in competitors:
            team = c.get("team") or {}
            name = team.get("displayName") or ""
            ha = (c.get("homeAway") or "").lower()
            if ha == "away":
                away = name
            elif ha == "home":
                home = name
    return away, home


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
            dict(team=team, period=period, clock=clock, shooter=shooter, result=result,
                 zone=zone, shot_value=val, pts=pts, description=text)
        )

    df = pd.DataFrame(rows)
    for c in schema:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df[schema]


def parse_shots_from_html_tables(summary_json: Dict[str, Any], tables: List[pd.DataFrame]) -> pd.DataFrame:
    schema = ["team", "period", "clock", "shooter", "result", "zone", "shot_value", "pts", "description"]
    away_name, home_name = teams_from_summary(summary_json)

    rows = []
    period_guess = 1

    for t in tables:
        if t is None or t.empty:
            continue

        cols = [str(c).strip().lower() for c in t.columns]
        df = t.copy()
        df.columns = cols

        clock_col = None
        for c in cols:
            if "time" in c or "clock" in c:
                clock_col = c
                break
        if clock_col is None:
            clock_col = cols[0]

        text_cols = [c for c in cols if c != clock_col]
        if len(text_cols) < 2:
            continue

        away_col, home_col = text_cols[0], text_cols[1]

        for _, r in df.iterrows():
            clock = r.get(clock_col, None)
            away_txt = r.get(away_col, None)
            home_txt = r.get(home_col, None)

            team = None
            text = None

            if isinstance(away_txt, str) and away_txt.strip():
                team = away_name or "Away"
                text = away_txt.strip()
            elif isinstance(home_txt, str) and home_txt.strip():
                team = home_name or "Home"
                text = home_txt.strip()

            if not text or not is_shot(text):
                continue

            result = parse_result(text)
            if not result:
                continue

            shooter = parse_shooter(text)
            val = shot_value(text)
            zone = infer_zone(text)
            pts = val if result == "Made" else 0

            rows.append(
                dict(team=team, period=period_guess, clock=clock, shooter=shooter, result=result,
                     zone=zone, shot_value=val, pts=pts, description=text)
            )

        period_guess += 1

    df_out = pd.DataFrame(rows)
    for c in schema:
        if c not in df_out.columns:
            df_out[c] = pd.Series(dtype="object")
    return df_out[schema]


def build_roster_headshots(team_id: str) -> pd.DataFrame:
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
    return pd.DataFrame(rows).drop_duplicates(subset=["player_norm"])


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


def score_value(x):
    if isinstance(x, dict):
        return x.get("displayValue") or x.get("value")
    return x


def parse_schedule(schedule_json: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    events = schedule_json.get("events", []) or []
    for e in events:
        eid = e.get("id")
        date = e.get("date")
        comp = (e.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", []) or []

        home = away = ""
        home_score = away_score = None
        for c in competitors:
            t = c.get("team") or {}
            ha = (c.get("homeAway") or "").lower()
            if ha == "home":
                home = t.get("displayName", "")
                home_score = score_value(c.get("score"))
            elif ha == "away":
                away = t.get("displayName", "")
                away_score = score_value(c.get("score"))

        if home and "Maryland" in home:
            home_away = "Home"
            opponent = away
        elif away and "Maryland" in away:
            home_away = "Away"
            opponent = home
        else:
            home_away = ""
            opponent = ""

        outcome = ""
        try:
            hs = int(home_score) if home_score is not None else None
            a_s = int(away_score) if away_score is not None else None
            if hs is not None and a_s is not None and home_away:
                maryland_score = hs if home_away == "Home" else a_s
                opp_score = a_s if home_away == "Home" else hs
                if maryland_score > opp_score:
                    outcome = "W"
                elif maryland_score < opp_score:
                    outcome = "L"
        except Exception:
            pass

        rows.append(
            {
                "game_id": str(eid) if eid else None,
                "date": date,
                "date_short": str(pd.to_datetime(date, errors="coerce").date()) if date else "",
                "opponent": opponent,
                "home_away": home_away,
                "outcome": outcome,
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["game_id"])
    return df


def fetch_and_parse_shots_for_game(game_id: str) -> Tuple[pd.DataFrame, str, int]:
    data = fetch_game_data(game_id)
    summary_json = data["summary"]
    pbp_json = data["pbp"]

    api_plays_found = len(extract_all_plays(pbp_json))
    shots_api = parse_shots_from_api(pbp_json)

    shots = shots_api
    source_used = "ESPN JSON API"
    if api_plays_found == 0 or len(shots_api) == 0:
        try:
            tables = fetch_pbp_html_tables(game_id)
            shots_html = parse_shots_from_html_tables(summary_json, tables)
            if len(shots_html) > 0:
                shots = shots_html
                source_used = "ESPN HTML (pandas.read_html fallback)"
        except Exception:
            pass

    return shots, source_used, api_plays_found


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def build_maryland_season_shots(team_id: str, team_name: str, max_games: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    schedule_json = fetch_team_schedule(team_id)
    sched = parse_schedule(schedule_json)
    sched_use = sched.head(max_games).copy()

    all_frames = []
    for _, r in sched_use.iterrows():
        gid = r["game_id"]
        shots_df, source_used, api_plays_found = fetch_and_parse_shots_for_game(gid)
        if shots_df is None or shots_df.empty:
            continue

        # ✅ Maryland-only
        shots_df = keep_only_team(shots_df, team_name)
        if shots_df is None or shots_df.empty:
            continue

        shots_df = shots_df.copy()
        shots_df["game_id"] = gid
        shots_df["date"] = r.get("date_short", "")
        shots_df["opponent"] = r.get("opponent", "")
        shots_df["home_away"] = r.get("home_away", "")
        shots_df["outcome"] = r.get("outcome", "")
        shots_df["source_used"] = source_used
        shots_df["api_plays_found"] = api_plays_found
        all_frames.append(shots_df)

    shots_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    return shots_all, sched


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
    if df is None or df.empty:
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
    if df_player is None or df_player.empty:
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


st.set_page_config(page_title="Maryland Shooting — ESPN", layout="wide")

refresh = st.button("Refresh data now (pull latest)")

st.title("College Basketball Shooting — ESPN Play-by-Play (Maryland Only)")

with st.sidebar:
    st.header("Roster (headshots)")
    roster_team_id = st.text_input("Roster teamId (Maryland = 120)", value=MARYLAND_TEAM_ID)
    roster_team_name = st.text_input("Roster team name", value=MARYLAND_NAME_DEFAULT)
    max_games = st.slider("How many schedule games to include", min_value=5, max_value=40, value=25, step=1)

    if refresh:
        fetch_game_data.clear()
        fetch_team_roster.clear()
        fetch_pbp_html_tables.clear()
        fetch_team_schedule.clear()
        build_maryland_season_shots.clear()

roster_lookup = build_roster_headshots(roster_team_id.strip())

with st.spinner("Loading Maryland schedule and aggregating Maryland shots across games..."):
    shots_all, sched = build_maryland_season_shots(roster_team_id.strip(), roster_team_name.strip(), max_games=max_games)

st.sidebar.subheader("Maryland schedule (from ESPN)")
if sched is not None and not sched.empty:
    show_sched = sched[["date_short", "opponent", "home_away", "outcome", "home_score", "away_score"]].copy()
    show_sched.rename(
        columns={"date_short": "Date", "opponent": "Opponent", "home_away": "H/A", "outcome": "W/L",
                 "home_score": "MD", "away_score": "Opp"},
        inplace=True,
    )
    st.sidebar.dataframe(show_sched, use_container_width=True, height=320, hide_index=True)

if shots_all is None or shots_all.empty:
    st.warning("No Maryland shots parsed from the selected schedule games. Try increasing games or click Refresh.")
    st.stop()

shots_all = attach_headshots(shots_all, roster_lookup)

with st.sidebar:
    st.subheader("Season filters (Maryland only)")
    game_opts = ["All"] + sorted(shots_all["game_id"].unique().tolist())
    game_pick = st.selectbox("Game", options=game_opts, index=0)

    opp_opts = ["All"] + sorted([x for x in shots_all["opponent"].dropna().unique().tolist() if str(x).strip()])
    opp_pick = st.selectbox("Opponent", options=opp_opts, index=0)

    zone_opts = ["All"] + sorted(shots_all["zone"].dropna().unique().tolist())
    zone_pick = st.selectbox("Zone", options=zone_opts, index=0)

    res_pick = st.selectbox("Result", options=["All", "Made", "Missed"], index=0)

    player_opts = sorted(shots_all["shooter"].dropna().unique().tolist())
    player_pick = st.selectbox("Player", options=player_opts, index=0 if player_opts else 0)

filt = shots_all.copy()
if game_pick != "All":
    filt = filt[filt["game_id"] == game_pick]
if opp_pick != "All":
    filt = filt[filt["opponent"] == opp_pick]
if zone_pick != "All":
    filt = filt[filt["zone"] == zone_pick]
if res_pick != "All":
    filt = filt[filt["result"] == res_pick]

player_df = filt[filt["shooter"] == player_pick].copy()

st.caption(f"Season shots (Maryland only): {len(shots_all)} | Filtered shots: {len(filt)} | Player shots: {len(player_df)}")

left, right = st.columns([1.5, 8.5])
with left:
    vals = player_df["headshot_url"].dropna().unique().tolist() if "headshot_url" in player_df.columns else []
    hs = vals[0] if vals else None
    if isinstance(hs, str) and hs.startswith("http"):
        st.image(hs, width=120)

with right:
    st.subheader(f"{player_pick} — Season Shooting Profile (Maryland Only)")
    pts, fg, three_pct, ft_pct = header_line(player_df)
    st.caption(f"PTS: {pts}  |  FG%: {fg*100:.1f}%  |  3P%: {three_pct*100:.1f}%  |  FT%: {ft_pct*100:.1f}%")
    st.caption("FG% color bands: Red < 30%, Yellow 30–40%, Green > 40%")

st.divider()

tab1, tab2, tab3 = st.tabs(["Zone breakdown", "Shot log", "Download"])
with tab1:
    zb = zone_breakdown(player_df).reset_index(drop=True)
    zb.insert(0, "", zb.index)
    st.dataframe(style_zone_table(zb), use_container_width=True, hide_index=True)

with tab2:
    cols = ["date", "opponent", "home_away", "outcome", "period", "clock", "shooter",
            "result", "zone", "shot_value", "pts", "description"]
    if "headshot_url" in player_df.columns:
        cols.append("headshot_url")
    st.dataframe(player_df[cols], use_container_width=True, hide_index=True)

with tab3:
    csv1 = filt.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered Maryland shots CSV", data=csv1, file_name="maryland_shots_filtered.csv")

    csv2 = shots_all.to_csv(index=False).encode("utf-8")
    st.download_button("Download full Maryland shots CSV", data=csv2, file_name="maryland_shots_full.csv")

st.caption("This dashboard is Maryland-only: it filters every game to Maryland shots before aggregating.")
