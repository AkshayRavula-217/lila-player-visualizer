import numpy as np
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


def get_fight_heatmap(df, map_choice, map_config, fight_type="All Kills"):

    if df is None or df.empty:
        return None, None, None

    df = _decode_events(df)

    # -----------------------------
    # UPDATED LOGIC (ONLY CHANGE)
    # -----------------------------
    fight_type = (fight_type or "All Kills").strip()

    if fight_type == "All Kills":
        # Only kill events (human + bot)
        combat_df = df[df["event"].isin(["Kill", "BotKill"])]

    elif fight_type == "Human vs Human Kills":
        combat_df = df[df["event"] == "Kill"]

    elif fight_type == "Human vs Bot Kills":
        combat_df = df[df["event"] == "BotKill"]

    elif fight_type == "All Combat Events":
        # Both kill and death locations
        combat_df = df[df["event"].isin(["Kill", "BotKill", "Killed", "KilledByBot"])]

    elif fight_type == "Human Deaths":
        combat_df = df[df["event"] == "Killed"]

    elif fight_type == "Deaths by Bot":
        combat_df = df[df["event"] == "KilledByBot"]

    else:
        combat_df = df[df["event"].isin(["Kill", "BotKill"])]

    # -----------------------------
    if combat_df.empty:
        return None, None, None

    cfg = map_config[map_choice]
    img_size = cfg.get("image_size", 1024)

    combat_df = world_to_pixel(combat_df, map_choice, map_config)

    combat_df = combat_df[
        (combat_df["pixel_x"] >= 0)
        & (combat_df["pixel_x"] <= img_size)
        & (combat_df["pixel_y"] >= 0)
        & (combat_df["pixel_y"] <= img_size)
    ]

    if combat_df.empty:
        return None, None, None

    bins = 144
    heatmap, x_edges, y_edges = np.histogram2d(
        combat_df["pixel_x"],
        combat_df["pixel_y"],
        bins=bins,
        range=[[0, img_size], [0, img_size]],
    )

    heatmap = gaussian_filter(heatmap, sigma=2)

    x = (x_edges[:-1] + x_edges[1:]) / 2
    y = (y_edges[:-1] + y_edges[1:]) / 2

    return heatmap, x, y


def get_fight_points(df, map_choice, map_config, fight_type="All Kills"):
    """
    Point representation of fights (for match-level drill-down).

    Returns a dataframe with pixel_x, pixel_y after applying the same
    fight_type categories as get_fight_heatmap.
    Assumes df is already globally filtered (including match_id).
    """
    if df is None or df.empty:
        return None

    df = _decode_events(df)

    fight_type = (fight_type or "All Kills").strip()

    if fight_type == "All Kills":
        combat_df = df[df["event"].isin(["Kill", "BotKill"])]
    elif fight_type == "Human vs Human Kills":
        combat_df = df[df["event"] == "Kill"]
    elif fight_type == "Human vs Bot Kills":
        combat_df = df[df["event"] == "BotKill"]
    elif fight_type == "All Combat Events":
        combat_df = df[df["event"].isin(["Kill", "BotKill", "Killed", "KilledByBot"])]
    elif fight_type == "Human Deaths":
        combat_df = df[df["event"] == "Killed"]
    elif fight_type == "Deaths by Bot":
        combat_df = df[df["event"] == "KilledByBot"]
    else:
        combat_df = df[df["event"].isin(["Kill", "BotKill"])]

    if combat_df.empty:
        return None

    cfg = map_config[map_choice]
    img_size = cfg.get("image_size", 1024)

    combat_df = world_to_pixel(combat_df, map_choice, map_config)

    combat_df = combat_df[
        (combat_df["pixel_x"] >= 0)
        & (combat_df["pixel_x"] <= img_size)
        & (combat_df["pixel_y"] >= 0)
        & (combat_df["pixel_y"] <= img_size)
    ]

    if combat_df.empty:
        return None

    return combat_df[["pixel_x", "pixel_y"]].reset_index(drop=True)


def get_kill_heatmap(df, MAP_CONFIG, map_choice, fight_type, IMG_WIDTH=None, IMG_HEIGHT=None):
    """
    Compatibility wrapper for the existing app.py imports.

    The main implementation lives in get_fight_heatmap().
    """
    # Map older UI labels to the current fight_type categories.
    ft = (fight_type or "").strip()
    if ft == "All Combat Events":
        ft_mapped = "All Combat Events"
    elif ft == "Human Fights":
        ft_mapped = "Human vs Human Kills"
    elif ft == "Bot Fights":
        ft_mapped = "Human vs Bot Kills"
    elif ft == "All Deaths":
        ft_mapped = "All Combat Events"  # closest; events are filtered inside get_fight_heatmap
    elif ft == "Human Deaths":
        ft_mapped = "Human Deaths"
    elif ft == "Deaths by Bot":
        ft_mapped = "Deaths by Bot"
    elif ft == "All Kills":
        ft_mapped = "All Kills"
    else:
        ft_mapped = "All Combat Events"

    return get_fight_heatmap(df, map_choice, MAP_CONFIG, fight_type=ft_mapped)