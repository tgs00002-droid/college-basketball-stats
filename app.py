# app.py
# NCAA Shot Zones Table + Player Headshots (ESPN)
# Uses TWO ESPN endpoints:
#  - playbyplay: to reliably get shot events
#  - summary: to get boxscore + athlete ids + headshots
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import re
from typing import Dict, Any, Tuple

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
# Helpers
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
    raise ValueError("Could not find a gameId. Paste an ESPN URL with /gameId/######### or just the numeric id.")


def _safe_get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_espn_game_data(game_id: str) -> Dict[str, Any]:
    """
    Pull both:
      - summary (boxscore + athletes + headshots)
      - playbyplay (all plays incl shots)
    """
    summary_candidates = [
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}",
        f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}",
    ]
    pbp_candidates = [
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/playbyplay?event={game_id}",
        f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/playbyplay?event={game_id}",
    ]

    last_err = None
    summary = None
    for url in summary_candidates:
        try:
            summary = _safe_get_json(url)
            break
        except Exception as e:
            last_err = e
    if summary is None:
        raise RuntimeError(f"Could not fetch SUMMARY JSON for game {game_id}. Last error: {last_err}")

    last_err = None
    pbp = None
    for url in pbp_candidates:
        try:
            pbp = _safe_get_json(url)
            break
        except Exception as e:
            last_err = e
    if pbp is None:
        raise RuntimeError(f"Could not fetch PLAY-BY-PLAY JSON for game {game_id}. Last error: {last_err}")

    return {"summary": summary, "pbp": pbp}


def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z\s\-']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def classify_zone(text: str) -> str:
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


def get_game_header(summary_json: Dict[str, Any]) -> Tuple[str, str]:
    try:
        comps = summary_json.get("header", {}).get("competitions", [])
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


def parse_shots_from_pbp(pbp_json: Dict[str, Any]) -> pd.DataFrame:
    plays = pbp_json.get("plays", [])
    if not isinstance(plays, list):
        plays = []

    rows = []
    for p in plays:
        text = (p.get("text") or "").strip()
        if not text or not is_shot_play(text):
            continue

        low = f" {text.lower()} "
        made = " made " in low
        missed = " missed " in low
        if not (made or missed):
            continue

        shooter = re.split(r"\bmade\b|\bmissed\b", text, flags=re.IGNORECASE)[0].strip().strip(".")
        zone = classify_zone(text)

        team = None
        if isinstance(p.get("team"), dict):
            team = p["team"].get("displayName") or p["team"].get("abbreviation")

        clock = None
        if isinstance(p.get("clock"), dict):
            clock = p["clock"].get("displayValue")

        period = None
        if isinstance(p.get("period"), dict):
            period = p["period"].get("number")

        rows.append(
            {
                "team": team,
                "period": period,
                "clock": clock,
                "shooter": shooter,
                "result": "Made" if made else "Missed",
                "zone": zone,
                "description": text,
            }
        )

    df = pd.DataFrame(rows)
    return df


def build_player_headshot_lookup(summary_json: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    box = summary_json.get("boxscore", {})
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
    shots = shots_df.copy()
    shots["shooter_norm"] = shots["shooter"].map(_norm_name)

    merged = shots.merge(
        lookup_df[["team", "player_norm", "headshot_url", "player"]],
        left_on=["team", "shooter_norm"],
        right_on=["team", "player_norm"],
        how="left",
    ).drop(columns=["player_norm"])

    # Fuzzy match fallback (helps if pbp uses slightly different naming)
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
        .agg(attempts=("attempts", "sum"), makes=("made", "sum"))
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
        .agg(attempts=("attempts", "sum"), makes=("made", "sum"))
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
    zone_filter = st.multiselect(
        "Zones",
        ["Rim", "Midrange", "3PT", "Free Throw", "Other"],
        default=["Rim", "Midrange", "3PT", "Free Throw"],
    )
    result_filter = st.multiselect("Result", ["Made", "Missed"], default=["Made", "Missed"])

    st.subheader("Export")
    export_csv = st.checkbox("Enable CSV downloads", value=True)

run = st.button("Load Game")

if run:
    try:
        game_id = extract_game_id(game_input)

        with st.spinner("Fetching ESPN summary + play-by-play..."):
            game_data = fetch_espn_game_data(game_id)

        summary_json = game_data["summary"]
        pbp_json = game_data["pbp"]

        away, home = get_game_header(summary_json)
        st.caption(f"Game ID: {game_id} | Away: {away} | Home: {home}")

        with st.spinner("Parsing shots from play-by-play..."):
            shots = parse_shots_from_pbp(pbp_json)

        if shots.empty:
            st.warning("Parsed 0 shots from play-by-play. ESPN may not be returning plays for this event.")
            st.stop()

        with st.spinner("Building headshot lookup from boxscore..."):
            lookup = build_player_headshot_lookup(summary_json)

        shots = attach_headshots(shots, lookup)

        # Drop incomplete rows + apply filters
        shots = shots.dropna(subset=["team", "shooter", "result", "zone"])
        shots = shots[shots["zone"].isin(zone_filter)]
        shots = shots[shots["result"].isin(result_filter)]

        # Layout
        left, right = st.columns([1.25, 1])

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

                # Sort: later periods first, clock descending within period
                def clock_to_seconds(c):
                    if not isinstance(c, str) or ":" not in c:
                        return -1
                    mm, ss = c.split(":")
                    try:
                        return int(mm) * 60 + int(ss)
                    except Exception:
                        return -1

                feed = shots.copy()
                feed["clock_sec"] = feed["clock"].map(clock_to_seconds)
                feed = feed.sort_values(["period", "clock_sec"], ascending=[True, False]).head(120)

                for _, r in feed.iterrows():
                    c1, c2 = st.columns([1, 6])
                    with c1:
                        url = r.get("headshot_url")
                        if isinstance(url, str) and url.startswith("http"):
                            st.image(url, width=56)
                    with c2:
                        st.write(f"{r['team']} | P{r['period']} {r['clock']} | {r['shooter']} | {r['result']} | {r['zone']}")
                        st.caption(r["description"])

        st.success("Done.")

    except Exception as e:
        st.error(str(e))
        st.info("Tip: paste a full ESPN play-by-play URL with /gameId/######### or just the gameId number.")

else:
    st.info("Click Load Game to build the shot-zone tables and headshot feed.")
