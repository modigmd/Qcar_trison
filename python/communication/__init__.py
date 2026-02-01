"""
Communication Module for QCar Autonomous Driving System

Handles IPC between Python (perception) and MATLAB (control):
- UDP communication (primary method)
- TCP communication (alternative)
- File-based communication (fallback)
"""

from .udp_comm import UDPCommunicator
from .message_protocol import (
    VisionMessage,
    ControlMessage,
    create_vision_message,
    parse_control_message
)

__all__ = [
    'UDPCommunicator',
    'VisionMessage',
    'ControlMessage',
    'create_vision_message',
    'parse_control_message',
]
