import pandas as pd

from Data.co_ordinate_mapper import world_to_pixel


def _decode_events(df):
    if "event" not in df.columns:
        return df
    df = df.copy()
    df["event"] = df["event"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
    )
    return df


def get_storm_deaths_points(df, map_choice, map_config):
    """
    Storm deaths metric (scatter, not heatmap).

    Returns a dict with separate dataframes for human and bot deaths,
    both already converted to pixel coordinates.

    Assumes global filters (map_id, match_id, timeline, player_type)
    are applied in the app.
    """
    result = {"human": None, "bot": None}

    if df is None or df.empty:
        return result

    df = _decode_events(df)
    if "event" not in df.columns:
        return result

    # Support both legacy and new naming
    storm_events = {
        "KilledByStorm",
        "killedbystorm",
        "StormDeath",
        "BotStormDeath",
    }

    storm_df = df[df["event"].astype(str).isin(storm_events)].copy()
    if storm_df.empty:
        return result

    # Map to human / bot categories
    def _is_bot_row(row):
        ev = str(row.get("event", ""))
        if ev == "BotStormDeath":
            return True
        if ev == "StormDeath":
            return False
        # Fallback to columns if present
        if "is_bot" in row:
            return bool(row["is_bot"])
        return False

    storm_df["is_bot_event"] = storm_df.apply(_is_bot_row, axis=1)

    storm_df = world_to_pixel(storm_df, map_choice, map_config)

    cfg = map_config[map_choice]
    img_size = cfg.get("image_size", 1024)
    storm_df = storm_df[
        (storm_df["pixel_x"] >= 0)
        & (storm_df["pixel_x"] <= img_size)
        & (storm_df["pixel_y"] >= 0)
        & (storm_df["pixel_y"] <= img_size)
    ]
    if storm_df.empty:
        return result

    human_df = storm_df[storm_df["is_bot_event"] == False][["pixel_x", "pixel_y"]]
    bot_df = storm_df[storm_df["is_bot_event"] == True][["pixel_x", "pixel_y"]]

    if not human_df.empty:
        result["human"] = human_df.reset_index(drop=True)
    if not bot_df.empty:
        result["bot"] = bot_df.reset_index(drop=True)

    return result


def get_storm_scatter(df, MAP_CONFIG, map_choice, IMG_WIDTH=None, IMG_HEIGHT=None):
    """
    Compatibility wrapper for the existing app.py.

    Returns a Plotly Figure containing storm death points (human=red, bot=blue).
    """
    import plotly.graph_objects as go

    points = get_storm_deaths_points(df, map_choice, MAP_CONFIG)
    fig = go.Figure()
    if not points:
        return fig

    human_pts = points.get("human")
    bot_pts = points.get("bot")

    if human_pts is not None and not human_pts.empty:
        fig.add_trace(
            go.Scattergl(
                x=human_pts["pixel_x"],
                y=human_pts["pixel_y"],
                mode="markers",
                marker=dict(color="red", size=6, opacity=0.7),
                name="Human Storm Death",
            )
        )

    if bot_pts is not None and not bot_pts.empty:
        fig.add_trace(
            go.Scattergl(
                x=bot_pts["pixel_x"],
                y=bot_pts["pixel_y"],
                mode="markers",
                marker=dict(color="blue", size=5, opacity=0.7),
                name="Bot Storm Death",
            )
        )

    return fig