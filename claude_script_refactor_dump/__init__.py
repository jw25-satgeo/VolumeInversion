"""Refactored, storage-independent analysis package."""
from .statistics import *
from .regression import ridge_fit
from .conditioning import conditioning_mask
from .profile_math import geometric_thickness, descale_profile, looks_whitened
from .scene_analysis import (
    summarize_fields, select_scenes, pooled_sums, pooled_cells,
    scene_statistics, image_mean_statistics, stratified_statistics,
    comparison_table, inter_reference_table, pooled_hist,
)
from .pipeline import analyze_fields, analyze_scenes
