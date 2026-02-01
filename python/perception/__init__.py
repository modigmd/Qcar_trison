"""
Perception Module for QCar Autonomous Driving System

This module contains computer vision components for:
- Traffic light detection (HSV-based)
- Person/pedestrian detection (YOLO-based)
- Lane detection (HSV-based)

Based on ODU AV Team architecture.
"""

from .traffic_light_detector import (
    TrafficLightDetector,
    TrafficLightDetection,
    TrafficLightState
)
from .person_detector import (
    PersonDetector,
    PersonDetection,
    create_detector as create_person_detector
)
from .lane_detector import (
    LaneDetector,
    LaneDetection
)

__all__ = [
    # Traffic Light
    'TrafficLightDetector',
    'TrafficLightDetection',
    'TrafficLightState',
    # Person Detection
    'PersonDetector',
    'PersonDetection',
    'create_person_detector',
    # Lane Detection
    'LaneDetector',
    'LaneDetection',
]
