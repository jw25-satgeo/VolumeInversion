"""Minimal analysis pipeline.

The pipeline begins with fields supplied by an external data layer. It performs no discovery,
loading, caching, downloading, or writing. Expensive field generation and persistence therefore
remain the responsibility of the existing data-management implementation.
"""
from .scene_analysis import summarize_fields


def analyze_fields(fields, scene=None, quantities=('h_geometric',), valids=None):
    """Analyze one already-materialized field set."""
    return summarize_fields(fields, scene=scene, quantities=quantities, valids=valids)


def analyze_scenes(scenes, quantities=('h_geometric',), valids=None):
    """Analyze an iterable of ``(scene_name, fields)`` pairs without retaining source fields."""
    return [
        summarize_fields(fields, scene=scene, quantities=quantities, valids=valids)
        for scene, fields in scenes
    ]
