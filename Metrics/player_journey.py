import re
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


def _is_human(uid):
    return bool(re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        str(uid), re.I
    ))


def get_player_journey(df, map_choice, map_config, user_id_filter=None):
    """
    Returns event dataframes for Player Rotation scatter view.

    user_id_filter is MANDATORY — filters to one specific player.
    match_id is OPTIONAL — already applied upstream in _apply_filters
    if provided, otherwise shows the player across all their matches.

    This gives a behaviour-centric view:
    - Without match_id: see how this player behaves across ALL their matches
    - With match_id:    drill into one specific match for this player

    NOTE: Timestamps are batch-flushed at match end in this dataset.
    Player Rotation shows WHERE the player went, not a time-ordered path.

    Returns dict with keys:
      human_pos   — Position events (this player, always human if UUID)
      bot_pos     — BotPosition events (bots in same match if no user filter)
      kills       — Kill + BotKill events
      deaths      — Killed + KilledByBot + KilledByStorm events
      loot        — all Loot events
      first_loot  — first Loot per match (landing zone proxy per match)
    """
    if df is None or df.empty:
        return None

    df = _decode_events(df)

    # Apply user_id filter — mandatory
    if user_id_filter and user_id_filter.strip():
        df = df[df["user_id"].astype(str).str.strip() == user_id_filter.strip()]

    if df.empty:
        return None

    cfg = map_config[map_choice]
    img_size = cfg.get("image_size", 1024)

    def _to_pixel(df_in):
        if df_in.empty:
            return df_in
        return world_to_pixel(df_in, map_choice, map_config)

    def _clip(df_in):
        if df_in.empty:
            return df_in
        return df_in[
            (df_in["pixel_x"] >= 0) & (df_in["pixel_x"] <= img_size) &
            (df_in["pixel_y"] >= 0) & (df_in["pixel_y"] <= img_size)
        ]

    def _prep(df_in):
        return _clip(_to_pixel(df_in.copy())) if not df_in.empty else df_in

    # Split by event type
    pos_df   = df[df["event"].isin(["Position", "BotPosition"])]
    kill_df  = df[df["event"].isin(["Kill", "BotKill"])]
    death_df = df[df["event"].isin(["Killed", "KilledByBot", "KilledByStorm"])]
    loot_df  = df[df["event"] == "Loot"]

    pos_df   = _prep(pos_df)
    kill_df  = _prep(kill_df)
    death_df = _prep(death_df)
    loot_df  = _prep(loot_df)

    # Split positions into human vs bot
    if not pos_df.empty and "user_id" in pos_df.columns:
        human_pos = pos_df[pos_df["user_id"].apply(_is_human)]
        bot_pos   = pos_df[~pos_df["user_id"].apply(_is_human)]
    else:
        human_pos = pos_df
        bot_pos   = pos_df.iloc[0:0]

    # First loot per match — landing zone proxy
    # Shows where the player lands in EACH match they played
    first_loot = pd.DataFrame()
    if not loot_df.empty and "user_id" in loot_df.columns and "ts" in loot_df.columns:
        group_cols = ["match_id"] if "match_id" in loot_df.columns else ["user_id"]
        first_loot = (
            loot_df.sort_values("ts")
            .drop_duplicates(subset=group_cols, keep="first")
        )

    return {
        "human_pos":  human_pos,
        "bot_pos":    bot_pos,
        "kills":      kill_df,
        "deaths":     death_df,
        "loot":       loot_df,
        "first_loot": first_loot,
    }


def get_players_in_match(df):
    """
    Returns lists of human and bot user_ids in the dataframe.
    Used to populate the player expander in the UI.
    """
    if df is None or df.empty or "user_id" not in df.columns:
        return [], []

    all_users = df["user_id"].astype(str).unique().tolist()
    humans = sorted([u for u in all_users if _is_human(u)])
    bots   = sorted([u for u in all_users if not _is_human(u)])
    return humans, bots