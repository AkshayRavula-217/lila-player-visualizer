"""
Compatibility wrapper for the existing app.py imports.

The main implementation lives in `landing_metrics.py`.
"""

from Metrics.landing_metrics import (
    get_landing_clusters_heatmap as _get_landing_clusters,
    get_landing_heatmap as _get_landing_heatmap,
)


def get_landing_heatmap(df, MAP_CONFIG, map_choice, IMG_WIDTH=None, IMG_HEIGHT=None):
    return _get_landing_heatmap(df, map_choice, MAP_CONFIG)


def get_landing_clusters(df, MAP_CONFIG, map_choice, IMG_WIDTH=None, IMG_HEIGHT=None):
    return _get_landing_clusters(df, map_choice, MAP_CONFIG)

