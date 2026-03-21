import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from Data.data_loader import load_data
from Data.co_ordinate_mapper import world_to_pixel
from Metrics.Kill_locations import get_fight_heatmap
from Metrics.Loot_locations import get_loot_heatmap
from Metrics.landing_metrics import get_landing_heatmap
from Metrics.storm_deaths import get_storm_deaths_points
from Metrics.endgame_position import get_endgame_heatmap
from Metrics.player_journey import get_player_journey, get_players_in_match
from map_config import MAP_CONFIG

st.set_page_config(layout="wide")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    metric = st.selectbox(
        "📊 Metric",
        [
            "Combat Zone",
            "Player Rotation",
            "Storm Deaths",
            "Starting Position",
            "Loot",
            "End Game Positions",
        ],
    )

    map_choice = st.selectbox("🗺 Map", list(MAP_CONFIG.keys()))

    date_choice = st.selectbox(
        "📅 Date",
        [
            "All Dates",
            "February_10",
            "February_11",
            "February_12",
            "February_13",
            "February_14",
        ],
    )

    match_id = st.text_input("🔍 Match ID (optional)")

    # User ID — mandatory for Player Journey, ignored for other metrics
    user_id_filter = st.text_input("👤 User ID (required for Player Rotation)")

    if metric != "Combat Zone":
        fight_type = None
    else:
        fight_type = st.selectbox(
            "⚔️ Combat Type",
            [
                "All Kills",
                "Human vs Human Kills",
                "Human vs Bot Kills",
                "All Combat Events",
                "Human Deaths",
                "Deaths by Bot",
            ],
        )

    # Human user_ids = UUID format (0019c582-...)
    # Bot user_ids   = numeric only (1388, 1411, ...)
    if metric == "Combat Zone":
        player_type = "All Players"
    else:
        player_type = st.selectbox(
            "👤 Player Type",
            ["All Players", "Humans Only", "Bots Only"],
        )

    apply = st.button("Apply", use_container_width=True)


# -----------------------------
# FILTER FUNCTION
# -----------------------------
def _apply_filters(df):
    df = df[df["map_id"] == map_choice]

    # Match ID filter
    if match_id:
        df = df[df["match_id"].astype(str).str.strip() == match_id.strip()]

    # Player type filter — UUID = human, numeric = bot
    # Note: not applied to Player Journey (handled inside that block)
    if metric != "Player Rotation":
        if player_type == "Humans Only":
            df = df[df["user_id"].astype(str).str.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                case=False
            )]
        elif player_type == "Bots Only":
            df = df[df["user_id"].astype(str).str.fullmatch(r"[0-9]+")]

    return df


# -----------------------------
# RUN
# -----------------------------
if apply:

    df = load_data("player_data", date_choice)

    if df is None or df.empty:
        st.stop()

    df = _apply_filters(df)

    if df.empty:
        st.warning("No data after filtering")
        st.stop()

    # Decode events
    df["event"] = df["event"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
    )

    # -----------------------------
    # MAP
    # -----------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(base_dir, MAP_CONFIG[map_choice]["image"])

    map_img = Image.open(map_path).convert("RGBA")
    size = MAP_CONFIG[map_choice].get("image_size", 1024)
    map_img = map_img.resize((size, size))

    map_arr = np.array(map_img)
    img_w, img_h = map_img.size

    fig = go.Figure()
    fig.add_trace(go.Image(z=map_arr, x0=0, y0=0, dx=1, dy=1))

    # -----------------------------
    # HEATMAP STYLE
    # -----------------------------
    heat_colorscale = [
        [0.0, "rgba(0,0,0,0)"],
        [0.1, "rgba(255,255,0,0.15)"],
        [0.3, "rgba(255,165,0,0.35)"],
        [0.6, "rgba(255,69,0,0.6)"],
        [1.0, "rgba(180,0,0,0.95)"],
    ]

    def _heatmap_zdata(heatmap, x, y):
        alpha = map_arr[:, :, 3]
        alpha_small = Image.fromarray(alpha).resize(
            (len(x), len(y)), Image.Resampling.NEAREST
        )
        mask = np.array(alpha_small) > 10
        return np.where(mask, heatmap.T, np.nan)

    def _to_pixel(df_in):
        if df_in.empty:
            return df_in
        return world_to_pixel(df_in, map_choice, MAP_CONFIG)

    def _clip(df_in):
        if df_in.empty:
            return df_in
        img_size = MAP_CONFIG[map_choice].get("image_size", 1024)
        return df_in[
            (df_in["pixel_x"] >= 0) & (df_in["pixel_x"] <= img_size) &
            (df_in["pixel_y"] >= 0) & (df_in["pixel_y"] <= img_size)
        ]

    # -----------------------------
    # PLAYER JOURNEY
    # Shows spatial coverage of all events for a match.
    # NOTE: Timestamps are batch-flushed at match end in this dataset —
    # this view shows WHERE players were, not a time-ordered path.
    # Enter a Match ID for a meaningful single-match view.
    # -----------------------------
    if metric == "Player Rotation":

        if not user_id_filter:
            st.warning("⚠️ User ID is required for Player Rotation. Enter a User ID in the sidebar.")
            st.stop()

        # Show matches for this specific user as a helper expander
        if "match_id" in df.columns and user_id_filter:
            user_df = df[df["user_id"].astype(str).str.strip() == user_id_filter.strip()]
            match_ids = user_df["match_id"].astype(str).unique().tolist()
            with st.expander(f"🎮 {len(match_ids)} match(es) found for this user"):
                for m in sorted(match_ids):
                    st.code(m, language=None)

        # Get all journey data from metric function
        journey = get_player_journey(df, map_choice, MAP_CONFIG, user_id_filter=user_id_filter)

        if journey is None:
            st.warning("No data found for this selection.")
            st.stop()

        human_pos  = journey["human_pos"]
        bot_pos    = journey["bot_pos"]
        kill_df    = journey["kills"]
        death_df   = journey["deaths"]
        loot_df    = journey["loot"]
        first_loot = journey["first_loot"]

        # Human positions — lime
        if not human_pos.empty:
            fig.add_trace(go.Scattergl(
                x=human_pos["pixel_x"], y=human_pos["pixel_y"],
                mode="markers",
                marker=dict(size=4, color="lime", opacity=0.5),
                name="Human Position",
            ))

        # Bot positions — gray
        if not bot_pos.empty:
            fig.add_trace(go.Scattergl(
                x=bot_pos["pixel_x"], y=bot_pos["pixel_y"],
                mode="markers",
                marker=dict(size=3, color="gray", opacity=0.3),
                name="Bot Position",
            ))

        # Kill events — red X
        if not kill_df.empty:
            fig.add_trace(go.Scattergl(
                x=kill_df["pixel_x"], y=kill_df["pixel_y"],
                mode="markers",
                marker=dict(size=9, color="red", opacity=0.9, symbol="x"),
                name="Kill",
            ))

        # Death events — blue X
        if not death_df.empty:
            fig.add_trace(go.Scattergl(
                x=death_df["pixel_x"], y=death_df["pixel_y"],
                mode="markers",
                marker=dict(size=9, color="dodgerblue", opacity=0.9, symbol="x"),
                name="Death",
            ))

        # Loot events — yellow diamond
        if not loot_df.empty:
            fig.add_trace(go.Scattergl(
                x=loot_df["pixel_x"], y=loot_df["pixel_y"],
                mode="markers",
                marker=dict(size=6, color="yellow", opacity=0.8, symbol="diamond"),
                name="Loot",
            ))

        # First loot per player — orange star (landing zone proxy)
        if not first_loot.empty:
            fig.add_trace(go.Scattergl(
                x=first_loot["pixel_x"], y=first_loot["pixel_y"],
                mode="markers",
                marker=dict(
                    size=14, color="orange", opacity=1.0,
                    symbol="star",
                    line=dict(color="white", width=1)
                ),
                name="First Loot — Landing Zone",
            ))

    # -----------------------------
    # FIGHT HEATMAP
    # Auto-switches to scatter when a match ID is entered
    # -----------------------------
    elif metric == "Combat Zone":

        if match_id:
            # Single match — scatter points are more meaningful than heatmap
            from Metrics.Kill_locations import get_fight_points
            pts = get_fight_points(df, map_choice, MAP_CONFIG, fight_type)
            if pts is not None and not pts.empty:
                fig.add_trace(go.Scattergl(
                    x=pts["pixel_x"], y=pts["pixel_y"],
                    mode="markers",
                    marker=dict(color="orange", size=7, opacity=0.85),
                    name="Fight Points",
                ))
        else:
            heatmap, x, y = get_fight_heatmap(df, map_choice, MAP_CONFIG, fight_type)
            if heatmap is not None:
                max_val = np.nanmax(heatmap)
                if max_val > 0:
                    heatmap = heatmap / max_val
                heatmap = np.power(heatmap, 0.25)
                z_data = _heatmap_zdata(heatmap, x, y)
                fig.add_trace(go.Heatmap(
                    z=z_data, x=x, y=y,
                    colorscale=heat_colorscale,
                    showscale=False, zsmooth="best", hoverongaps=False,
                ))

    # -----------------------------
    # LOOT HEATMAP
    # Auto-switches to scatter when a match ID is entered
    # -----------------------------
    elif metric == "Loot":

        if match_id:
            from Metrics.Loot_locations import get_loot_points
            pts = get_loot_points(df, map_choice, MAP_CONFIG, fight_type)
            if pts is not None and not pts.empty:
                fig.add_trace(go.Scattergl(
                    x=pts["pixel_x"], y=pts["pixel_y"],
                    mode="markers",
                    marker=dict(color="yellow", size=7, opacity=0.85, symbol="diamond"),
                    name="Loot Points",
                ))
        else:
            heatmap, x, y = get_loot_heatmap(df, map_choice, MAP_CONFIG, fight_type)
            if heatmap is not None:
                max_val = np.nanmax(heatmap)
                if max_val > 0:
                    heatmap = heatmap / max_val
                heatmap = np.power(heatmap, 0.25)
                z_data = _heatmap_zdata(heatmap, x, y)
                fig.add_trace(go.Heatmap(
                    z=z_data, x=x, y=y,
                    colorscale=heat_colorscale,
                    showscale=False, zsmooth="best", hoverongaps=False,
                ))

    # -----------------------------
    # LANDING HEATMAP
    # Auto-switches to scatter when a match ID is entered
    # -----------------------------
    elif metric == "Starting Position":

        if match_id:
            # Show first loot per player as scatter for single match
            loot_df = df[df["event"] == "Loot"].copy()
            if not loot_df.empty and "ts" in loot_df.columns:
                first_loot = (
                    loot_df.sort_values("ts")
                    .drop_duplicates(subset=["user_id"], keep="first")
                )
                first_loot = _clip(_to_pixel(first_loot))
                if not first_loot.empty:
                    fig.add_trace(go.Scattergl(
                        x=first_loot["pixel_x"], y=first_loot["pixel_y"],
                        mode="markers",
                        marker=dict(
                            color="orange", size=12, opacity=1.0,
                            symbol="star", line=dict(color="white", width=1)
                        ),
                        name="Landing Points",
                    ))
        else:
            heatmap, x, y = get_landing_heatmap(df, map_choice, MAP_CONFIG)
            if heatmap is not None:
                max_val = np.nanmax(heatmap)
                if max_val > 0:
                    heatmap = heatmap / max_val
                heatmap = np.power(heatmap, 0.25)
                z_data = _heatmap_zdata(heatmap, x, y)
                fig.add_trace(go.Heatmap(
                    z=z_data, x=x, y=y,
                    colorscale=heat_colorscale,
                    showscale=False, zsmooth="best", hoverongaps=False,
                ))

    # -----------------------------
    # ENDGAME POSITIONS HEATMAP
    # -----------------------------
    elif metric == "End Game Positions":

        endgame_player = "bot" if player_type == "Bots Only" else \
                         "human" if player_type == "Humans Only" else "all"
        heatmap, x, y = get_endgame_heatmap(df, map_choice, MAP_CONFIG, player_type=endgame_player)
        if heatmap is not None:
            max_val = np.nanmax(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            heatmap = np.power(heatmap, 0.25)
            z_data = _heatmap_zdata(heatmap, x, y)
            fig.add_trace(go.Heatmap(
                z=z_data, x=x, y=y,
                colorscale=heat_colorscale,
                showscale=False, zsmooth="best", hoverongaps=False,
            ))
        else:
            st.warning("No endgame position data available for this selection.")

    # -----------------------------
    # STORM DEATHS SCATTER
    # -----------------------------
    elif metric == "Storm Deaths":

        pts = get_storm_deaths_points(df, map_choice, MAP_CONFIG)
        if pts:
            if pts.get("human") is not None:
                fig.add_trace(go.Scattergl(
                    x=pts["human"]["pixel_x"], y=pts["human"]["pixel_y"],
                    mode="markers",
                    marker=dict(color="red", size=6, opacity=0.7),
                    name="Human",
                ))
            if pts.get("bot") is not None:
                fig.add_trace(go.Scattergl(
                    x=pts["bot"]["pixel_x"], y=pts["bot"]["pixel_y"],
                    mode="markers",
                    marker=dict(color="blue", size=5, opacity=0.7),
                    name="Bot",
                ))

    # -----------------------------
    # LAYOUT
    # -----------------------------
    fig.update_xaxes(range=[0, img_w], visible=False)
    fig.update_yaxes(range=[0, img_h], autorange="reversed", visible=False)

    fig.update_layout(
        height=900,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)