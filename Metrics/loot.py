"""
Compatibility wrapper for the existing app.py imports.

The main implementation lives in `Loot_locations.py`.
"""

from Metrics.Loot_locations import get_loot_heatmap as _get_loot_heatmap
from Metrics.Loot_locations import get_loot_points as _get_loot_points


def get_loot_heatmap(df, map_choice, MAP_CONFIG, fight_type="All Fights"):
    return _get_loot_heatmap(df, map_choice, MAP_CONFIG, fight_type=fight_type)


def get_loot_points(df, map_choice, MAP_CONFIG, fight_type="All Fights"):
    return _get_loot_points(df, map_choice, MAP_CONFIG, fight_type=fight_type)

