"""
Message Protocol for Python <-> MATLAB Communication

Defines the message formats for IPC between:
- Python (perception/vision)
- MATLAB (control)

Messages are JSON-encoded for simplicity and debugging.
"""

import json
import struct
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from enum import Enum


class VehicleState(Enum):
    """Vehicle state enumeration matching MATLAB state machine."""
    CRUISE = 0
    STOP_RED = 1
    STOP_PERSON = 2
    PICKUP_WAIT = 3
    STOP_SIGN = 4


@dataclass
class VisionMessage:
    """
    Message sent from Python (vision) to MATLAB (control).

    Contains perception outputs that inform control decisions.
    """
    # Timestamp (seconds since start)
    timestamp: float

    # Traffic light detection
    red_light: bool
    green_light: bool
    yellow_light: bool
    traffic_light_confidence: float

    # Person detection
    person_detected: bool
    person_in_pickup_zone: bool
    person_confidence: float

    # Lane detection
    lane_detected: bool
    steering_suggestion: float  # [-1, 1] suggested steering from lane following
    lane_center_offset: float  # pixels from center

    # Suggested state (Python's recommendation based on perception)
    suggested_state: int  # VehicleState value

    # Debug info
    frame_id: int

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))

    def to_bytes(self) -> bytes:
        """Convert to bytes for UDP transmission."""
        return self.to_json().encode('utf-8')

    @classmethod
    def from_json(cls, json_str: str) -> 'VisionMessage':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'VisionMessage':
        """Create from bytes."""
        return cls.from_json(data.decode('utf-8'))


@dataclass
class ControlMessage:
    """
    Message sent from MATLAB (control) to Python (optional feedback).

    Contains control state and acknowledgments.
    """
    # Timestamp
    timestamp: float

    # Current vehicle state
    current_state: int  # VehicleState value

    # Control outputs (for visualization)
    throttle: float
    steering: float
    speed: float

    # Position (for debugging)
    x: float
    y: float
    heading: float

    # Acknowledgment
    last_vision_frame_id: int

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))

    def to_bytes(self) -> bytes:
        """Convert to bytes for UDP transmission."""
        return self.to_json().encode('utf-8')

    @classmethod
    def from_json(cls, json_str: str) -> 'ControlMessage':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ControlMessage':
        """Create from bytes."""
        return cls.from_json(data.decode('utf-8'))


def create_vision_message(
    timestamp: float,
    red_light: bool = False,
    green_light: bool = False,
    yellow_light: bool = False,
    traffic_light_confidence: float = 0.0,
    person_detected: bool = False,
    person_in_pickup_zone: bool = False,
    person_confidence: float = 0.0,
    lane_detected: bool = False,
    steering_suggestion: float = 0.0,
    lane_center_offset: float = 0.0,
    suggested_state: int = VehicleState.CRUISE.value,
    frame_id: int = 0
) -> VisionMessage:
    """
    Factory function to create a VisionMessage.

    Args:
        All fields of VisionMessage

    Returns:
        VisionMessage instance
    """
    return VisionMessage(
        timestamp=timestamp,
        red_light=red_light,
        green_light=green_light,
        yellow_light=yellow_light,
        traffic_light_confidence=traffic_light_confidence,
        person_detected=person_detected,
        person_in_pickup_zone=person_in_pickup_zone,
        person_confidence=person_confidence,
        lane_detected=lane_detected,
        steering_suggestion=steering_suggestion,
        lane_center_offset=lane_center_offset,
        suggested_state=suggested_state,
        frame_id=frame_id
    )


def parse_control_message(data: bytes) -> Optional[ControlMessage]:
    """
    Parse control message from bytes.

    Args:
        data: Raw bytes from UDP

    Returns:
        ControlMessage or None if parsing fails
    """
    try:
        return ControlMessage.from_bytes(data)
    except Exception as e:
        print(f"Error parsing control message: {e}")
        return None


# Binary protocol for lower latency (alternative)
class BinaryProtocol:
    """
    Binary protocol for lower latency communication.

    Format:
    - Header: 4 bytes (magic number)
    - Length: 4 bytes (uint32, little endian)
    - Payload: variable length JSON

    This is an alternative to plain JSON for production use.
    """

    MAGIC = b'QCAR'

    @staticmethod
    def pack(message: bytes) -> bytes:
        """Pack message with header."""
        length = len(message)
        return BinaryProtocol.MAGIC + struct.pack('<I', length) + message

    @staticmethod
    def unpack(data: bytes) -> Optional[bytes]:
        """Unpack message, verify header."""
        if len(data) < 8:
            return None

        if data[:4] != BinaryProtocol.MAGIC:
            return None

        length = struct.unpack('<I', data[4:8])[0]

        if len(data) < 8 + length:
            return None

        return data[8:8+length]


# Example usage
if __name__ == "__main__":
    # Create and serialize a vision message
    msg = create_vision_message(
        timestamp=1.0,
        red_light=True,
        green_light=False,
        traffic_light_confidence=0.85,
        person_detected=True,
        person_in_pickup_zone=True,
        person_confidence=0.72,
        lane_detected=True,
        steering_suggestion=0.05,
        suggested_state=VehicleState.STOP_RED.value,
        frame_id=42
    )

    # Serialize
    json_str = msg.to_json()
    print(f"JSON: {json_str}")

    # Deserialize
    msg2 = VisionMessage.from_json(json_str)
    print(f"Parsed: {msg2}")

    # Binary protocol test
    packed = BinaryProtocol.pack(msg.to_bytes())
    print(f"Packed length: {len(packed)}")

    unpacked = BinaryProtocol.unpack(packed)
    print(f"Unpacked: {VisionMessage.from_bytes(unpacked)}")
