# app.py
# Maryland (or any NCAA) Shot Location Table + Player Headshots
# Data source: ESPN public game summary JSON (no API key needed)
#
# Run:
#   pip install streamlit pandas requests rapidfuzz
#   streamlit run app.py
#
# Notes:
# - ESPN play-by-play text does NOT always include true X/Y shot coordinates.
#   This app builds "where on the court" via zone inference from shot descriptions:
#   Rim, Midrange, 3PT, Free Throw, Other.
# - Player headshots are pulled from ESPN athlete IDs in the boxscore payload.

import re
from typing import Dict, Any, Optional, Tuple

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
DEFAULT_GAME_URL = "https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401822757"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"


# -----------------------------
# Utilities
# -----------------------------
def extract_game_id(espn_url_or_id: str) -> str:
    s = (espn_url_or_id or "").strip()
    if s.isdigit():
        return s
    m = re.search(r"gameId/(\d+)", s)
    if m:
        return m.group(1)
    m = re.search(r"event=(\d+)", s)
    if m:
        return m.group(1)
    raise ValueError("Could not find a gameId in that input. Paste an ESPN URL with /gameId/######### or just the numeric id.")


def _safe_get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_espn_summary_json(game_id: str) -> Dict[str, Any]:
    """
    ESPN uses a few hosts for the same API path. Try multiple.
    """
    candidates = [
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}",
        f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}",
    ]
    last_err = None
    for url in candidates:
        try:
            return _safe_get_json(url)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not fetch ESPN summary JSON for game {game_id}. Last error: {last_err}")


def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z\s\-']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def classify_zone(text: str) -> str:
    """
    Zone inference from play text.
    You can expand this mapping later.
    """
    t = (text or "").lower()

    if "free throw" in t:
        return "Free Throw"
    if "three point" in t or "3-point" in t or "3 point" in t:
        return "3PT"
    if any(k in t for k in ["dunk", "layup", "tip-in", "putback", "alley-oop"]):
        return "Rim"
    if any(k in t for k in ["jumper", "fadeaway", "hook shot", "pullup", "pull-up"]):
        return "Midrange"
    return "Other"


def is_shot_play(text: str) -> bool:
    t = (text or "").lower()
    if not any(k in t for k in ["made", "missed"]):
        return False
    return any(k in t for k in ["jumper", "three point", "layup", "dunk", "free throw", "tip-in", "hook shot", "putback"])


def find_plays_list(game_json: Dict[str, Any]) -> list:
    """
    ESPN summary payload usually has a top-level 'plays' list.
    If not, we scan common nested keys.
    """
    if isinstance(game_json.get("plays"), list):
        return game_json["plays"]

    # Common alternate nesting
    for key in ["pbp", "playByPlay", "gamecast", "gameCast"]:
        v = game_json.get(key)
        if isinstance(v, dict) and isinstance(v.get("plays"), list):
            return v["plays"]

    # Shallow scan: find first dict that has 'plays'
    for v in game_json.values():
        if isinstance(v, dict) and isinstance(v.get("plays"), list):
            return v["plays"]

    raise RuntimeError("Could not locate a plays list in the ESPN JSON. ESPN payload format may have changed.")


def get_game_header(game_json: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (away_team, home_team) display names if available.
    """
    try:
        comps = game_json.get("header", {}).get("competitions", [])
        if comps:
            competitors = comps[0].get("competitors", [])
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_name = away.get("team", {}).get("displayName") if away else "Away"
            home_name = home.get("team", {}).get("displayName") if home else "Home"
            return away_name or "Away", home_name or "Home"
    except Exception:
        pass
    return "Away", "Home"


def parse_shots(game_json: Dict[str, Any]) -> pd.DataFrame:
    plays = find_plays_list(game_json)
    rows = []

    for p in plays:
        text = (p.get("text") or "").strip()
        if not text or not is_shot_play(text):
            continue

        low = f" {text.lower()} "
        made = " made " in low
        missed = " missed " in low

        # Shooter text is usually before 'made'/'missed'
        shooter = re.split(r"\bmade\b|\bmissed\b", text, flags=re.IGNORECASE)[0].strip().strip(".")

        zone = classify_zone(text)

        # team info is often present as p["team"]["displayName"] or abbreviation
        team = None
        if isinstance(p.get("team"), dict):
            team = p["team"].get("displayName") or p["team"].get("abbreviation")

        # clock/period
        clock = p.get("clock", {}).get("displayValue") if isinstance(p.get("clock"), dict) else p.get("clock")
        period = p.get("period", {}).get("number") if isinstance(p.get("period"), dict) else p.get("period")

        rows.append(
            {
                "team": team,
                "period": period,
                "clock": clock,
                "shooter": shooter,
                "result": "Made" if made else ("Missed" if missed else None),
                "zone": zone,
                "description": text,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No shots parsed from play-by-play. ESPN may not be returning shot text in this payload.")
    return df


def build_player_headshot_lookup(game_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Builds player -> headshot_url lookup from boxscore.
    ESPN often includes athlete.id and headshot.href.
    If headshot missing but id exists, we construct a standard ESPN headshot URL.
    """
    rows = []
    box = game_json.get("boxscore", {})
    players_blocks = box.get("players", [])

    for team_block in players_blocks:
        team_obj = team_block.get("team") or {}
        team_name = team_obj.get("displayName") or team_obj.get("abbreviation")

        for stat_group in team_block.get("statistics", []):
            for athlete_row in stat_group.get("athletes", []):
                athlete = athlete_row.get("athlete") or {}
                name = athlete.get("displayName") or athlete.get("shortName")
                athlete_id = athlete.get("id")

                headshot = athlete.get("headshot")
                if isinstance(headshot, dict):
                    headshot = headshot.get("href")

                if (not headshot) and athlete_id:
                    headshot = f"https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/{athlete_id}.png"

                if name and headshot:
                    rows.append(
                        {
                            "team": team_name,
                            "player": name,
                            "player_norm": _norm_name(name),
                            "athlete_id": athlete_id,
                            "headshot_url": headshot,
                        }
                    )

    df = pd.DataFrame(rows).drop_duplicates(subset=["team", "player_norm"])
    return df


def attach_headshots(shots_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach headshot_url to each shot based on (team + shooter name).
    Uses exact normalized match first.
    Optionally uses fuzzy match if rapidfuzz is installed.
    """
    shots = shots_df.copy()
    shots["shooter_norm"] = shots["shooter"].map(_norm_name)

    # Exact join: team + normalized shooter name
    merged = shots.merge(
        lookup_df[["team", "player_norm", "headshot_url", "player"]],
        left_on=["team", "shooter_norm"],
        right_on=["team", "player_norm"],
        how="left",
    ).drop(columns=["player_norm"])

    # If missing headshots and rapidfuzz is available, do fuzzy matching within each team
    if RAPIDFUZZ_AVAILABLE:
        missing = merged["headshot_url"].isna()
        if missing.any():
            filled = merged.copy()
            for team_name in filled.loc[missing, "team"].dropna().unique():
                team_lookup = lookup_df[lookup_df["team"] == team_name]
                choices = list(team_lookup["player_norm"].unique())

                if not choices:
                    continue

                idxs = filled.index[(filled["team"] == team_name) & (filled["headshot_url"].isna())].tolist()
                for i in idxs:
                    target = filled.at[i, "shooter_norm"]
                    if not target:
                        continue
                    match = process.extractOne(target, choices, scorer=fuzz.WRatio)
                    if match and match[1] >= 88:
                        best_norm = match[0]
                        row = team_lookup[team_lookup["player_norm"] == best_norm].head(1)
                        if not row.empty:
                            filled.at[i, "headshot_url"] = row["headshot_url"].iloc[0]
                            filled.at[i, "player"] = row["player"].iloc[0]
            merged = filled

    return merged


def zone_summary(df_shots: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_shots.assign(
            made=(df_shots["result"] == "Made").astype(int),
            attempts=1,
        )
        .groupby(["team", "zone"], as_index=False)
        .agg(
            attempts=("attempts", "sum"),
            makes=("made", "sum"),
        )
    )
    out["misses"] = out["attempts"] - out["makes"]
    out["fg_pct"] = (out["makes"] / out["attempts"]).round(3)
    return out.sort_values(["team", "zone"])


def player_summary(df_shots: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_shots.assign(
            made=(df_shots["result"] == "Made").astype(int),
            attempts=1,
        )
        .groupby(["team", "shooter"], as_index=False)
        .agg(
            attempts=("attempts", "sum"),
            makes=("made", "sum"),
        )
    )
    out["misses"] = out["attempts"] - out["makes"]
    out["fg_pct"] = (out["makes"] / out["attempts"]).round(3)
    return out.sort_values(["team", "attempts"], ascending=[True, False])


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NCAA Shot Zones + Headshots", layout="wide")

st.title("NCAA Shot Zones Table with Player Headshots")

with st.sidebar:
    st.subheader("Game Input")
    game_input = st.text_input("Paste ESPN play-by-play URL or gameId", value=DEFAULT_GAME_URL)
    show_shot_log = st.checkbox("Show shot log table", value=True)
    show_zone_table = st.checkbox("Show zone summary table", value=True)
    show_player_table = st.checkbox("Show player summary table", value=True)
    show_feed = st.checkbox("Show shot feed with faces", value=True)

    st.subheader("Filters")
    zone_filter = st.multiselect("Zones", ["Rim", "Midrange", "3PT", "Free Throw", "Other"], default=["Rim", "Midrange", "3PT", "Free Throw"])
    result_filter = st.multiselect("Result", ["Made", "Missed"], default=["Made", "Missed"])

    st.subheader("Export")
    export_csv = st.checkbox("Enable CSV downloads", value=True)

run = st.button("Load Game")

if run:
    try:
        game_id = extract_game_id(game_input)
        with st.spinner("Fetching ESPN data..."):
            game_json = fetch_espn_summary_json(game_id)

        away, home = get_game_header(game_json)
        st.caption(f"Game ID: {game_id} | Away: {away} | Home: {home}")

        with st.spinner("Parsing shots..."):
            shots = parse_shots(game_json)

        # Headshots lookup
        with st.spinner("Building headshot lookup..."):
            lookup = build_player_headshot_lookup(game_json)

        # Attach headshots
        shots = attach_headshots(shots, lookup)

        # Clean / filter
        shots = shots.dropna(subset=["team", "shooter", "result"])
        shots = shots[shots["zone"].isin(zone_filter)]
        shots = shots[shots["result"].isin(result_filter)]

        # Layout
        left, right = st.columns([1.2, 1])

        with left:
            if show_zone_table:
                st.subheader("Where are they shooting from (Zone Summary)")
                zs = zone_summary(shots)
                st.dataframe(zs, use_container_width=True, hide_index=True)

                if export_csv and not zs.empty:
                    st.download_button(
                        "Download zone_summary.csv",
                        data=zs.to_csv(index=False).encode("utf-8"),
                        file_name="zone_summary.csv",
                        mime="text/csv",
                    )

            if show_player_table:
                st.subheader("Player Shooting Summary")
                ps = player_summary(shots)
                st.dataframe(ps, use_container_width=True, hide_index=True)

                if export_csv and not ps.empty:
                    st.download_button(
                        "Download player_summary.csv",
                        data=ps.to_csv(index=False).encode("utf-8"),
                        file_name="player_summary.csv",
                        mime="text/csv",
                    )

            if show_shot_log:
                st.subheader("Shot Log (with inferred court zone)")
                cols = ["team", "period", "clock", "shooter", "result", "zone", "description", "headshot_url"]
                st.dataframe(shots[cols], use_container_width=True, hide_index=True)

                if export_csv and not shots.empty:
                    st.download_button(
                        "Download shots_log.csv",
                        data=shots[cols].to_csv(index=False).encode("utf-8"),
                        file_name="shots_log.csv",
                        mime="text/csv",
                    )

        with right:
            if show_feed:
                st.subheader("Shot Feed (faces + readable)")
                shots_feed = shots.sort_values(["period", "clock"], ascending=[True, False]).head(80)

                for _, r in shots_feed.iterrows():
                    c1, c2 = st.columns([1, 6])
                    with c1:
                        if pd.notna(r.get("headshot_url")) and str(r["headshot_url"]).startswith("http"):
                            st.image(r["headshot_url"], width=56)
                        else:
                            st.write("")
                    with c2:
                        st.write(
                            f"{r['team']} | P{r['period']} {r['clock']} | {r['shooter']} | {r['result']} | {r['zone']}"
                        )
                        st.caption(r["description"])

        st.success("Done.")

    except Exception as e:
        st.error(str(e))
        st.info(
            "If this specific game payload is missing plays or boxscore athletes, try a different gameId. "
            "ESPN sometimes changes the JSON shape per event."
        )

else:
    st.info("Click Load Game to build the shot-zone tables and headshot feed.")

