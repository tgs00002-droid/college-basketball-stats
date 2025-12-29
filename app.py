# app.py
# NCAA Shot Zones Table + Player Headshots (ESPN)
# Robust to ESPN JSON differences + verb tense changes

import re
from typing import Dict, Any, List, Tuple

import pandas as pd
import requests
import streamlit as st

# Optional fuzzy matching
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
DEFAULT_GAME_URL = "https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401817514"
USER_AGENT = "Mozilla/5.0"


# -----------------------------
# ESPN helpers
# -----------------------------
def extract_game_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if s.isdigit():
        return s
    m = re.search(r"gameId/(\d+)", s)
    if m:
        return m.group(1)
    raise ValueError("Could not find gameId in input.")


def _safe_get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_game_data(game_id: str) -> Dict[str, Any]:
    summary = _safe_get_json(
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}"
    )
    pbp = _safe_get_json(
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/playbyplay?event={game_id}"
    )
    return {"summary": summary, "pbp": pbp}


# -----------------------------
# Parsing logic
# -----------------------------
def extract_all_plays(pbp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    plays: List[Dict[str, Any]] = []

    if isinstance(pbp_json.get("plays"), list):
        plays.extend(pbp_json["plays"])

    for d in pbp_json.get("drives", []) or []:
        if isinstance(d.get("plays"), list):
            plays.extend(d["plays"])

    for p in pbp_json.get("periods", []) or []:
        if isinstance(p.get("plays"), list):
            plays.extend(p["plays"])

    return plays


def is_shot_play(text: str) -> bool:
    t = (text or "").lower()
    if not any(k in t for k in ["made", "missed", "makes", "misses"]):
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
        ]
    )


def classify_zone(text: str) -> str:
    t = (text or "").lower()
    if "free throw" in t:
        return "Free Throw"
    if "three point" in t or "three-point" in t:
        return "3PT"
    if any(k in t for k in ["layup", "dunk", "tip-in", "putback"]):
        return "Rim"
    if "jumper" in t or "hook shot" in t:
        return "Midrange"
    return "Other"


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_shots_from_pbp(pbp_json: Dict[str, Any]) -> pd.DataFrame:
    plays = extract_all_plays(pbp_json)
    rows = []

    for p in plays:
        if not isinstance(p, dict):
            continue

        text = (p.get("text") or "").strip()
        if not is_shot_play(text):
            continue

        low = f" {text.lower()} "
        made = any(k in low for k in [" made ", " makes "])
        missed = any(k in low for k in [" missed ", " misses "])

        shooter = re.split(
            r"\bmade\b|\bmissed\b|\bmakes\b|\bmisses\b",
            text,
            flags=re.IGNORECASE,
        )[0].strip().strip(".")

        zone = classify_zone(text)

        team = None
        if isinstance(p.get("team"), dict):
            team = p["team"].get("displayName")

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

    return pd.DataFrame(rows)


def build_headshot_lookup(summary_json: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for team_block in summary_json.get("boxscore", {}).get("players", []):
        team_name = team_block.get("team", {}).get("displayName")

        for stat_group in team_block.get("statistics", []):
            for a in stat_group.get("athletes", []):
                athlete = a.get("athlete", {})
                name = athlete.get("displayName")
                aid = athlete.get("id")

                if name and aid:
                    rows.append(
                        {
                            "team": team_name,
                            "player_norm": normalize_name(name),
                            "headshot_url": f"https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/{aid}.png",
                        }
                    )

    return pd.DataFrame(rows).drop_duplicates()


def attach_headshots(shots: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
    shots["player_norm"] = shots["shooter"].map(normalize_name)

    merged = shots.merge(
        lookup,
        left_on=["team", "player_norm"],
        right_on=["team", "player_norm"],
        how="left",
    )

    return merged


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("NCAA Shot Zones Table with Player Headshots")

with st.sidebar:
    game_input = st.text_input("Paste ESPN play-by-play URL or gameId", DEFAULT_GAME_URL)
    run = st.button("Load Game")

if run:
    try:
        game_id = extract_game_id(game_input)
        data = fetch_game_data(game_id)

        shots = parse_shots_from_pbp(data["pbp"])
        lookup = build_headshot_lookup(data["summary"])
        shots = attach_headshots(shots, lookup)

        st.caption(f"Shots parsed: {len(shots)}")

        if shots.empty:
            st.warning("No shots found.")
            st.stop()

        st.subheader("Shot Log")
        st.dataframe(
            shots[
                [
                    "team",
                    "period",
                    "clock",
                    "shooter",
                    "result",
                    "zone",
                    "description",
                    "headshot_url",
                ]
            ],
            use_container_width=True,
        )

        st.subheader("Shot Feed")
        for _, r in shots.head(100).iterrows():
            c1, c2 = st.columns([1, 6])
            with c1:
                if isinstance(r["headshot_url"], str):
                    st.image(r["headshot_url"], width=55)
            with c2:
                st.write(
                    f"{r['team']} | P{r['period']} {r['clock']} | {r['shooter']} | {r['result']} | {r['zone']}"
                )
                st.caption(r["description"])

    except Exception as e:
        st.error(str(e))
