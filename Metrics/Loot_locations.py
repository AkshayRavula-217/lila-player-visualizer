import numpy as np
from Data.co_ordinate_mapper import world_to_pixel


def _decode_event_col(df):
    if "event" not in df.columns:
        return df
    df = df.copy()
    df["event"] = df["event"].apply(
        lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
    )
    return df


def get_loot_heatmap(df, map_choice, map_config, fight_type="All Fights"):

    required_cols = {"x", "z", "event"}
    if df is None or df.empty or not required_cols.issubset(set(df.columns)):
        return None, None, None

    df = _decode_event_col(df)

    # -----------------------------
    # ONLY LOOT EVENTS
    # -----------------------------
    loot_df = df[df["event"].astype(str) == "Loot"].copy()
    if loot_df.empty:
        return None, None, None

    # -----------------------------
    # ONLY DEATH EVENTS (FIXED)
    # -----------------------------
    death_events = {"Killed", "KilledByBot"}
    death_df = df[df["event"].astype(str).isin(death_events)].copy()

    cfg = map_config[map_choice]
    img_size = cfg.get("image_size", 1024)

    # Convert to pixel
    loot_df = world_to_pixel(loot_df, map_choice, map_config)

    if not death_df.empty:
        death_df = world_to_pixel(death_df, map_choice, map_config)

    # Remove out-of-bounds
    loot_df = loot_df[
        (loot_df["pixel_x"] >= 0)
        & (loot_df["pixel_x"] <= img_size)
        & (loot_df["pixel_y"] >= 0)
        & (loot_df["pixel_y"] <= img_size)
    ]

    if loot_df.empty:
        return None, None, None

    # If no deaths → return all loot
    if death_df.empty or "ts" not in loot_df.columns or "ts" not in death_df.columns:
        bins = 144
        heatmap, x_edges, y_edges = np.histogram2d(
            loot_df["pixel_x"],
            loot_df["pixel_y"],
            bins=bins,
            range=[[0, img_size], [0, img_size]],
        )
        x = (x_edges[:-1] + x_edges[1:]) / 2
        y = (y_edges[:-1] + y_edges[1:]) / 2
        return heatmap, x, y

    # ---------------------------------------------
    # REMOVE LOOT FROM DEATH DROPS
    # ---------------------------------------------
    loot_df["_rx"] = np.rint(loot_df["pixel_x"]).astype("int32")
    loot_df["_ry"] = np.rint(loot_df["pixel_y"]).astype("int32")

    death_df["_rx"] = np.rint(death_df["pixel_x"]).astype("int32")
    death_df["_ry"] = np.rint(death_df["pixel_y"]).astype("int32")

    # Get earliest death per pixel per match
    death_core = (
        death_df[["match_id", "_rx", "_ry", "ts"]]
        .dropna()
        .groupby(["match_id", "_rx", "_ry"], as_index=False)["ts"]
        .min()
        .rename(columns={"ts": "death_ts"})
    )

    loot_keys = loot_df[["match_id", "_rx", "_ry", "ts"]].dropna()

    merged = loot_keys.merge(
        death_core,
        on=["match_id", "_rx", "_ry"],
        how="left",
    )

    # Remove loot AFTER death
    merged["_is_death_drop"] = merged["death_ts"].notna() & (
        merged["death_ts"] < merged["ts"]
    )

    invalid_keys = merged[merged["_is_death_drop"]][
        ["match_id", "_rx", "_ry", "ts"]
    ]

    if not invalid_keys.empty:
        invalid_keys["_key"] = (
            invalid_keys["match_id"].astype(str)
            + "_"
            + invalid_keys["_rx"].astype(str)
            + "_"
            + invalid_keys["_ry"].astype(str)
            + "_"
            + invalid_keys["ts"].astype(str)
        )

        loot_df["_key"] = (
            loot_df["match_id"].astype(str)
            + "_"
            + loot_df["_rx"].astype(str)
            + "_"
            + loot_df["_ry"].astype(str)
            + "_"
            + loot_df["ts"].astype(str)
        )

        loot_df = loot_df[~loot_df["_key"].isin(set(invalid_keys["_key"]))]

    if loot_df.empty:
        return None, None, None

    # -----------------------------
    # HEATMAP
    # -----------------------------
    bins = 144
    heatmap, x_edges, y_edges = np.histogram2d(
        loot_df["pixel_x"],
        loot_df["pixel_y"],
        bins=bins,
        range=[[0, img_size], [0, img_size]],
    )

    x = (x_edges[:-1] + x_edges[1:]) / 2
    y = (y_edges[:-1] + y_edges[1:]) / 2

    return heatmap, x, y


def get_loot_points(df, map_choice, map_config, fight_type="All Fights"):
    """
    Loot scatter (instead of heatmap).

    Applies the same "exclude likely death-drop loot" logic as get_loot_heatmap,
    then returns remaining loot positions as pixel coordinates for plotting.
    """
    required_cols = {"x", "z", "event"}
    if df is None or df.empty or not required_cols.issubset(set(df.columns)):
        return None

    fight_type = (fight_type or "All Fights").strip()

    # Decode event (bytes -> str)
    df = _decode_event_col(df)

    loot_df = df[df["event"].astype(str) == "Loot"].copy()
    if loot_df.empty:
        return None

    combat_events = None
    if fight_type == "All Kills":
        combat_events = {"Kill", "BotKill"}
    elif fight_type == "Human Kills":
        combat_events = {"Kill"}
    elif fight_type == "Bot Kills":
        combat_events = {"BotKill"}
    elif fight_type == "Human Fights":
        # Older UI label: treat as human-caused kills only
        combat_events = {"Kill"}
    elif fight_type == "Bot Fights":
        # Older UI label: treat as bot-caused kills only
        combat_events = {"BotKill"}
    elif fight_type == "All Fights":
        combat_events = {"Kill", "BotKill", "Killed", "KilledByBot", "BotKilled"}
    elif fight_type == "All Deaths":
        combat_events = {"Killed", "KilledByBot", "BotKilled"}
    elif fight_type == "Player Deaths":
        combat_events = {"Killed"}
    elif fight_type == "Bot-caused Deaths":
        combat_events = {"KilledByBot", "BotKilled"}
    else:
        combat_events = {"Kill", "BotKill", "Killed", "KilledByBot", "BotKilled"}

    # If we don't have match_id, we can't do death-drop time filtering.
    if "match_id" not in df.columns:
        cfg = map_config[map_choice]
        img_size = cfg.get("image_size", 1024)
        loot_df = world_to_pixel(loot_df, map_choice, map_config)
        loot_df = loot_df[
            (loot_df["pixel_x"] >= 0)
            & (loot_df["pixel_x"] <= img_size)
            & (loot_df["pixel_y"] >= 0)
            & (loot_df["pixel_y"] <= img_size)
        ]
        if loot_df.empty:
            return None
        return loot_df[["pixel_x", "pixel_y"]].reset_index(drop=True)

    combat_df = df[df["event"].astype(str).isin(combat_events)].copy()

    cfg = map_config[map_choice]
    img_size = cfg.get("image_size", 1024)

    loot_df = world_to_pixel(loot_df, map_choice, map_config)
    if not combat_df.empty:
        combat_df = world_to_pixel(combat_df, map_choice, map_config)

    # Remove out-of-bounds early
    loot_df = loot_df[
        (loot_df["pixel_x"] >= 0)
        & (loot_df["pixel_x"] <= img_size)
        & (loot_df["pixel_y"] >= 0)
        & (loot_df["pixel_y"] <= img_size)
    ]
    if loot_df.empty:
        return None

    if combat_df.empty or "ts" not in loot_df.columns or "ts" not in combat_df.columns:
        return loot_df[["pixel_x", "pixel_y"]].reset_index(drop=True)

    # ---------------------------------------------
    # Exclude death-drop loot (same-ish pixel, death_ts < loot_ts)
    # ---------------------------------------------
    loot_df["_rx"] = np.rint(loot_df["pixel_x"]).astype("int32")
    loot_df["_ry"] = np.rint(loot_df["pixel_y"]).astype("int32")
    combat_df["_rx"] = np.rint(combat_df["pixel_x"]).astype("int32")
    combat_df["_ry"] = np.rint(combat_df["pixel_y"]).astype("int32")

    combat_core = (
        combat_df[["match_id", "_rx", "_ry", "ts"]]
        .dropna(subset=["match_id", "_rx", "_ry", "ts"])
        .groupby(["match_id", "_rx", "_ry"], as_index=False)["ts"]
        .min()
        .rename(columns={"ts": "death_ts"})
    )

    loot_keys = loot_df[["match_id", "_rx", "_ry", "ts"]].dropna(
        subset=["match_id", "_rx", "_ry", "ts"]
    )

    merged = loot_keys.merge(
        combat_core,
        on=["match_id", "_rx", "_ry"],
        how="left",
    )

    merged["_is_death_drop"] = merged["death_ts"].notna() & (
        merged["death_ts"] < merged["ts"]
    )

    death_drop_keys = merged[merged["_is_death_drop"]][
        ["match_id", "_rx", "_ry", "ts"]
    ]

    if not death_drop_keys.empty:
        death_drop_keys["_key"] = (
            death_drop_keys["match_id"].astype(str)
            + "_"
            + death_drop_keys["_rx"].astype(str)
            + "_"
            + death_drop_keys["_ry"].astype(str)
            + "_"
            + death_drop_keys["ts"].astype(str)
        )
        loot_df["_key"] = (
            loot_df["match_id"].astype(str)
            + "_"
            + loot_df["_rx"].astype(str)
            + "_"
            + loot_df["_ry"].astype(str)
            + "_"
            + loot_df["ts"].astype(str)
        )
        death_keys_set = set(death_drop_keys["_key"])
        loot_df = loot_df[~loot_df["_key"].isin(death_keys_set)]

    if loot_df.empty:
        return None

    return loot_df[["pixel_x", "pixel_y"]].reset_index(drop=True)
