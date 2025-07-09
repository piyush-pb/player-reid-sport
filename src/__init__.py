# Player Re-Identification System
# Liat.ai AI Intern Assignment

"""
Player Re-Identification System Package

This package contains the core modules for player detection, tracking, and re-identification.

Modules:
- detector: Player detection using YOLOv11
- tracker: Multi-object tracking with re-identification
- visualizer: Visualization and plotting utilities
- pipeline: Main orchestration pipeline
"""

__version__ = "1.0.0"
__author__ = "AI Intern"
__email__ = "intern@liat.ai"

from .detector import PlayerDetector
from .tracker import PlayerTracker, PlayerTrack
from .visualizer import VideoVisualizer, ResultsVisualizer
from .pipeline import PlayerReIDPipeline

__all__ = [
    'PlayerDetector',
    'PlayerTracker', 
    'PlayerTrack',
    'VideoVisualizer',
    'ResultsVisualizer',
    'PlayerReIDPipeline'
] 