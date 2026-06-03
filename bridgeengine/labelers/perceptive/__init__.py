"""Perception comparison labelers.

These wrappers preserve the pre-pivot LEWM-derived SAM/depth/track artifacts
as an available comparison baseline. They are not part of the main pi0.7-style
rich-prompt benchmark.
"""

from __future__ import annotations

from bridgeengine.labelers.depth import DepthLabeler
from bridgeengine.labelers.masks import MaskLabeler
from bridgeengine.labelers.tracks import TrackLabeler

__all__ = ["DepthLabeler", "MaskLabeler", "TrackLabeler"]
