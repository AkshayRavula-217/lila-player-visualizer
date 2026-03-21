def world_to_pixel(*args, **kwargs):
    """
    Convert world coordinates to pixel coordinates.

    Supports two call styles:
      1) world_to_pixel(x, z, origin_x, origin_z, scale, img_width, img_height) -> (pixel_x, pixel_y)
      2) world_to_pixel(df, map_choice, map_config) -> df with added pixel_x/pixel_y columns
    """

    # Style 2: (df, map_choice, map_config)
    if len(args) == 3 and not kwargs:
        df, map_choice, map_config = args
        cfg = map_config[map_choice]

        origin_x = cfg["origin_x"]
        origin_z = cfg["origin_z"]
        scale = cfg["scale"]
        img_size = cfg.get("image_size", 1024)

        df = df.copy()
        df["pixel_x"], df["pixel_y"] = world_to_pixel(
            df["x"],
            df["z"],
            origin_x,
            origin_z,
            scale,
            img_size,
            img_size,
        )
        return df

    # Style 1: (x, z, origin_x, origin_z, scale, img_width, img_height)
    if len(args) == 7 and not kwargs:
        x, z, origin_x, origin_z, scale, img_width, img_height = args
        pixel_x = ((x - origin_x) / scale) * img_width
        pixel_y = (1 - (z - origin_z) / scale) * img_height
        return pixel_x, pixel_y

    raise TypeError(
        "world_to_pixel expected (df, map_choice, map_config) or "
        "(x, z, origin_x, origin_z, scale, img_width, img_height)"
    )


# ------------------------------------------------------
# OPTIONAL: Reverse mapping (useful for debugging / future)
# ------------------------------------------------------

def pixel_to_world(pixel_x, pixel_y, origin_x, origin_z, scale, img_width, img_height):
    """
    Convert pixel coordinates back → world coordinates
    """

    x = (pixel_x / img_width) * scale + origin_x
    z = ((1 - (pixel_y / img_height)) * scale) + origin_z

    return x, z