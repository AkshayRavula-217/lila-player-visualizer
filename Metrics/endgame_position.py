import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from Data.co_ordinate_mapper import world_to_pixel


def _decode_events(df):
    if "event" not in df.columns:
        return df
    df = df.copy()
    df["event"] = df["event"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
    )
    return df


def _get_endgame_points(df, player_type="all"):
    """
    Endgame proxy = the single last kill per match.

    Per match, sort kill events by ts descending and take only the
    very last kill. This pinpoints the exact final fight location.
    """
    if df is None or df.empty:
        return df

    needed = {"x", "z", "match_id", "ts"}
    if not needed.issubset(set(df.columns)):
        return df.iloc[0:0]

    df = _decode_events(df)

    if "event" not in df.columns:
        return df.iloc[0:0]

    if player_type == "human":
        kill_events  = ["Kill"]
        death_events = ["Killed"]
    elif player_type == "bot":
        kill_events  = ["BotKill"]
        death_events = ["BotKilled"]
    else:
        kill_events  = ["Kill", "BotKill"]
        death_events = ["Killed", "BotKilled"]
    fight_events = kill_events + death_events

    df_fights = df[df["event"].isin(fight_events)].copy()

    if df_fights.empty:
        return df.iloc[0:0]

    results = []
    for match_id, group in df_fights.groupby("match_id"):
        group_sorted = group.sort_values("ts", ascending=False)

        # Only the single last kill per match = true endgame location
        last_kill = group_sorted[group_sorted["event"].isin(kill_events)].head(1)

        if not last_kill.empty:
            results.append(last_kill)

    if not results:
        return df.iloc[0:0]

    return pd.concat(results, ignore_index=True)


def get_endgame_heatmap(df, map_choice, map_config, player_type="all"):
    """
    Heatmap of final kill locations per match.
    Uses the single last kill per match as endgame proxy.
    """
    if df is None or df.empty:
        return None, None, None

    endgame = _get_endgame_points(df, player_type=player_type)

    if endgame is None or endgame.empty:
        return None, None, None

    cfg = map_config[map_choice]
    img_size = cfg.get("image_size", 1024)

    endgame = world_to_pixel(endgame, map_choice, map_config)

    endgame = endgame[
        (endgame["pixel_x"] >= 0)
        & (endgame["pixel_x"] <= img_size)
        & (endgame["pixel_y"] >= 0)
        & (endgame["pixel_y"] <= img_size)
    ]

    if endgame.empty:
        return None, None, None

    bins = 144
    heatmap, x_edges, y_edges = np.histogram2d(
        endgame["pixel_x"],
        endgame["pixel_y"],
        bins=bins,
        range=[[0, img_size], [0, img_size]],
    )

    heatmap = gaussian_filter(heatmap, sigma=2)

    x = (x_edges[:-1] + x_edges[1:]) / 2
    y = (y_edges[:-1] + y_edges[1:]) / 2

    return heatmap, x, y