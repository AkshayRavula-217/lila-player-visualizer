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


def _get_landing_points(df):
    """
    Use first observed position per (match_id, user_id) as landing proxy.
    (i.e., take the earliest `ts` row for each player within each match).
    """
    if df is None or df.empty:
        return df

    # We only need x/z/ts and match/user grouping for landing proxy.
    needed = {"x", "z", "match_id", "user_id"}
    if not needed.issubset(set(df.columns)):
        return df.iloc[0:0]

    # Decode event if present (harmless, but keeps output consistent)
    df = _decode_events(df)

    if "ts" not in df.columns:
        # No timestamp: fall back to first row per (match_id, user_id)
        landing = (
            df.sort_values(["match_id", "user_id"])
            .drop_duplicates(subset=["match_id", "user_id"], keep="first")
        )
    else:
        # Earliest timestamped position per player within each match
        landing = (
            df.sort_values(["match_id", "user_id", "ts"])
            .drop_duplicates(subset=["match_id", "user_id"], keep="first")
        )

    return landing


def _landing_heatmap_core(df, map_choice, map_config, sigma):
    if df is None or df.empty:
        return None, None, None

    cfg = map_config[map_choice]
    img_size = cfg.get("image_size", 1024)

    df = world_to_pixel(df, map_choice, map_config)

    df = df[
        (df["pixel_x"] >= 0)
        & (df["pixel_x"] <= img_size)
        & (df["pixel_y"] >= 0)
        & (df["pixel_y"] <= img_size)
    ]
    if df.empty:
        return None, None, None

    # Consistent bin count for all heatmaps
    bins = 144
    heatmap, x_edges, y_edges = np.histogram2d(
        df["pixel_x"],
        df["pixel_y"],
        bins=bins,
        range=[[0, img_size], [0, img_size]],
    )

    heatmap = gaussian_filter(heatmap, sigma=sigma)

    x = (x_edges[:-1] + x_edges[1:]) / 2
    y = (y_edges[:-1] + y_edges[1:]) / 2

    return heatmap, x, y


def get_landing_heatmap(df, map_choice, map_config):
    """
    Landing heatmap using first loot event per player as landing proxy.

    Assumes global filters have already been applied.
    """
    landing = _get_landing_points(df)
    return _landing_heatmap_core(landing, map_choice, map_config, sigma=2)


def get_landing_clusters_heatmap(df, map_choice, map_config):
    """
    Landing clusters: same landing points as landing heatmap but with
    stronger smoothing for region-level patterns.
    """
    landing = _get_landing_points(df)
    return _landing_heatmap_core(landing, map_choice, map_config, sigma=6)

