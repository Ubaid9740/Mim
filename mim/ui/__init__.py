"""Tactical Observability UI subsystem for Mim: TacticalTheme, EyeSentinelWidget, and ScrubberOverlayWindow."""

from mim.ui.theme import TacticalTheme, apply_window_capture_cloaking
from mim.ui.eye_sentinel import EyeSentinelWidget
from mim.ui.scrubber_overlay import ScrubberOverlayWindow

__all__ = [
    "TacticalTheme",
    "apply_window_capture_cloaking",
    "EyeSentinelWidget",
    "ScrubberOverlayWindow",
]

