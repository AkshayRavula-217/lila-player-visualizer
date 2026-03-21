from Data.co_ordinate_mapper import world_to_pixel


def get_match_timeline(df, map_choice, map_config, max_time):
    """
    Returns all events up to selected time
    """

    if df is None or df.empty:
        return None

    df = df[df["ts"] <= max_time].copy()

    if df.empty:
        return None

    df = world_to_pixel(df, map_choice, map_config)

    df = df.dropna(subset=["pixel_x", "pixel_y"])

    return df