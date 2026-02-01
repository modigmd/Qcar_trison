"""
Main Vision Module for QCar Autonomous Driving System

Integrates all perception components and state machine:
- Traffic light detection (HSV)
- Person detection (YOLO)
- Lane detection (HSV)
- State machine (CRUISE, STOP_RED, STOP_PERSON, PICKUP_WAIT)
- UDP communication with MATLAB

Based on ODU AV Team architecture.

Author: QCar AV System
"""

import cv2
import numpy as np
import time
import yaml
import argparse
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import perception modules
from perception import (
    TrafficLightDetector,
    TrafficLightDetection,
    TrafficLightState,
    create_person_detector,
    PersonDetection,
    LaneDetector,
    LaneDetection
)

# Import communication modules
from communication import (
    UDPCommunicator,
    VisionMessage,
    create_vision_message
)
from communication.message_protocol import VehicleState

# Import QLabs camera system
from qlabs_camera import QCarCameraSystem

# Import vehicle interface
from qcar_vehicle import create_vehicle, LEDState


class VisionStateMachine:
    """
    State machine for vehicle control decisions.

    States:
    - CRUISE: Normal driving, following lane
    - STOP_RED: Stopped at red traffic light
    - STOP_PERSON: Stopped for pedestrian pickup
    - PICKUP_WAIT: Waiting during pickup (timer-based)
    """

    def __init__(self, config: dict = None):
        """
        Initialize the state machine.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        sm_config = self.config.get('state_machine', {})

        # State
        self.current_state = VehicleState.CRUISE
        self.previous_state = VehicleState.CRUISE

        # Timing
        self.state_entry_time = time.time()
        self.pickup_wait_time = sm_config.get('pickup_wait_time', 3.0)
        self.green_confirm_frames = sm_config.get('green_confirm_frames', 5)

        # Counters for temporal filtering
        self.green_count = 0
        self.red_count = 0
        self.person_count = 0

        # Thresholds
        self.confirm_threshold = 5

    def update(self,
               traffic_light: TrafficLightDetection,
               person: PersonDetection,
               lane: LaneDetection) -> VehicleState:
        """
        Update state machine based on perception inputs.

        Args:
            traffic_light: Traffic light detection result
            person: Person detection result
            lane: Lane detection result

        Returns:
            Current vehicle state
        """
        self.previous_state = self.current_state

        # Update counters with temporal filtering
        if traffic_light.state == TrafficLightState.RED:
            self.red_count = min(self.red_count + 1, self.confirm_threshold * 2)
            self.green_count = max(self.green_count - 1, 0)
        elif traffic_light.state == TrafficLightState.GREEN:
            self.green_count = min(self.green_count + 1, self.confirm_threshold * 2)
            self.red_count = max(self.red_count - 1, 0)

        if person.in_pickup_zone:
            self.person_count = min(self.person_count + 1, self.confirm_threshold * 2)
        else:
            self.person_count = max(self.person_count - 1, 0)

        # State transitions
        if self.current_state == VehicleState.CRUISE:
            # Check for red light
            if self.red_count >= self.confirm_threshold:
                self._transition_to(VehicleState.STOP_RED)

            # Check for person in pickup zone
            elif self.person_count >= self.confirm_threshold:
                self._transition_to(VehicleState.STOP_PERSON)

        elif self.current_state == VehicleState.STOP_RED:
            # Wait for green light
            if self.green_count >= self.green_confirm_frames:
                self._transition_to(VehicleState.CRUISE)

        elif self.current_state == VehicleState.STOP_PERSON:
            # Transition to pickup wait after stopping
            elapsed = time.time() - self.state_entry_time
            if elapsed > 0.5:  # Brief stop before waiting
                self._transition_to(VehicleState.PICKUP_WAIT)

        elif self.current_state == VehicleState.PICKUP_WAIT:
            # Wait for pickup timer
            elapsed = time.time() - self.state_entry_time
            if elapsed >= self.pickup_wait_time:
                self._transition_to(VehicleState.CRUISE)
                # Reset person counter to avoid immediate re-detection
                self.person_count = 0

        return self.current_state

    def _transition_to(self, new_state: VehicleState):
        """Transition to a new state."""
        if new_state != self.current_state:
            print(f"State transition: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.state_entry_time = time.time()

    def get_state_name(self) -> str:
        """Get current state name."""
        return self.current_state.name

    def get_time_in_state(self) -> float:
        """Get time spent in current state."""
        return time.time() - self.state_entry_time


class QLabsCamera:
    """
    Camera interface for QLabs virtual environment.

    Wraps the QCarCameraSystem to provide camera frames.
    Uses qvl library (QLabsQCar2.get_image) for QLabs integration.

    NO WEBCAM FALLBACK - if QLabs fails, Python stops immediately.
    """

    def __init__(self, config: dict = None, use_webcam: bool = False):
        """
        Initialize camera interface.

        Args:
            config: Configuration dictionary
            use_webcam: DEPRECATED - webcam fallback is disabled

        Raises:
            RuntimeError: If use_webcam=True (no longer supported)
        """
        self.config = config or {}

        if use_webcam:
            raise RuntimeError(
                "Webcam mode is disabled. This system requires QLabs. "
                "Remove --webcam flag and ensure QLabs is running with QCar spawned."
            )

        # Use the QCarCameraSystem (strict mode, no fallback)
        self.camera_system = QCarCameraSystem(config=config, use_mock=False)
        self.frame_count = 0

    def start(self) -> bool:
        """
        Start camera capture.

        Returns:
            True if successful

        Raises:
            RuntimeError: If QLabs camera initialization fails
        """
        # This will raise RuntimeError if QLabs is not available
        self.camera_system.initialize()
        print("QLabs camera started successfully")
        return True

    def stop(self):
        """Stop camera capture."""
        self.camera_system.terminate()

    def get_frame(self, camera_name: str = 'front') -> Optional[np.ndarray]:
        """
        Get a frame from the specified camera.

        Args:
            camera_name: Camera name (ignored - uses configured camera)

        Returns:
            BGR image (OpenCV format)

        Raises:
            RuntimeError: If camera read fails
        """
        # qvl returns BGR directly, no conversion needed
        frame = self.camera_system.get_frame(camera_name)

        if frame is not None:
            self.frame_count += 1

        return frame

    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """
        Get frames from all cameras.

        Returns:
            Dictionary of camera_name -> BGR image
        """
        return self.camera_system.get_all_frames()

    def get_frame_count(self) -> int:
        """Get total frames captured."""
        return self.frame_count


class VisionSystem:
    """
    Main vision system integrating all components.

    Manages:
    - Camera capture
    - Perception (traffic light, person, lane)
    - State machine
    - UDP communication with MATLAB
    - Debug visualization
    """

    def __init__(self, config_path: str = None, use_webcam: bool = False):
        """
        Initialize the vision system.

        Args:
            config_path: Path to configuration YAML file
            use_webcam: Use webcam instead of QLabs (for testing)
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.camera = QLabsCamera(self.config, use_webcam)
        self.traffic_light_detector = TrafficLightDetector(self.config)
        self.person_detector = create_person_detector(self.config)
        self.lane_detector = LaneDetector(self.config)
        self.state_machine = VisionStateMachine(self.config)
        self.communicator = UDPCommunicator(self.config)

        # Timing
        self.start_time = 0.0
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = 0.0
        self.fps_frame_count = 0

        # Debug settings
        debug_config = self.config.get('debug', {})
        self.show_camera = debug_config.get('show_camera_feed', True)
        self.show_detections = debug_config.get('show_detections', True)
        self.show_steering = debug_config.get('show_steering_info', True)
        self.show_state = debug_config.get('show_state_info', True)

        # Windows
        self.window_name = "QCar Vision System"

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        # Try default path
        default_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if default_path.exists():
            with open(default_path, 'r') as f:
                return yaml.safe_load(f)

        print("Warning: No config file found, using defaults")
        return {}

    def start(self) -> bool:
        """
        Start the vision system.

        Returns:
            True if successful

        Raises:
            RuntimeError: If QLabs camera fails to initialize
        """
        print("Starting QCar Vision System...")
        print("=" * 50)

        # Start camera (will raise RuntimeError if QLabs not available)
        try:
            self.camera.start()
        except RuntimeError as e:
            print("")
            print("ERROR: Failed to start QLabs camera")
            print(f"  {e}")
            print("")
            print("This system requires QLabs with a spawned QCar.")
            print("Run test_qlabs_camera.py first to verify setup.")
            raise

        # Start communication
        if not self.communicator.start():
            print("Warning: Failed to start UDP communication")
            # Continue anyway for testing

        self.start_time = time.time()
        self.last_fps_time = time.time()

        print("=" * 50)
        print("Vision system started successfully")
        print("=" * 50)
        return True

    def stop(self):
        """Stop the vision system."""
        print("Stopping QCar Vision System...")
        self.camera.stop()
        self.communicator.stop()
        cv2.destroyAllWindows()
        print("Vision system stopped")

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame through all perception components.

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (traffic_light, person, lane, state)
        """
        # Run perception
        traffic_light = self.traffic_light_detector.detect_with_temporal_filter(frame)
        person = self.person_detector.detect(frame)
        lane = self.lane_detector.detect(frame)

        # Update state machine
        state = self.state_machine.update(traffic_light, person, lane)

        return traffic_light, person, lane, state

    def create_vision_message(self,
                               traffic_light: TrafficLightDetection,
                               person: PersonDetection,
                               lane: LaneDetection,
                               state: VehicleState) -> VisionMessage:
        """Create a vision message for MATLAB communication."""
        return create_vision_message(
            timestamp=time.time() - self.start_time,
            red_light=(traffic_light.state == TrafficLightState.RED),
            green_light=(traffic_light.state == TrafficLightState.GREEN),
            yellow_light=(traffic_light.state == TrafficLightState.YELLOW),
            traffic_light_confidence=traffic_light.confidence,
            person_detected=person.detected,
            person_in_pickup_zone=person.in_pickup_zone,
            person_confidence=person.confidence,
            lane_detected=(lane.left_line_detected or lane.right_line_detected),
            steering_suggestion=lane.steering_suggestion,
            lane_center_offset=lane.lane_center_offset,
            suggested_state=state.value,
            frame_id=self.frame_count
        )

    def visualize(self,
                  frame: np.ndarray,
                  traffic_light: TrafficLightDetection,
                  person: PersonDetection,
                  lane: LaneDetection,
                  state: VehicleState) -> np.ndarray:
        """
        Create visualization with all detections.

        Args:
            frame: Input BGR frame
            traffic_light: Traffic light detection
            person: Person detection
            lane: Lane detection
            state: Current vehicle state

        Returns:
            Visualization frame
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # Draw lane detection
        if self.show_detections:
            output = self.lane_detector.visualize(output, lane)

        # Draw traffic light detection
        if self.show_detections and traffic_light.bounding_box:
            x, y, bw, bh = traffic_light.bounding_box
            if traffic_light.state == TrafficLightState.RED:
                color = (0, 0, 255)
            elif traffic_light.state == TrafficLightState.GREEN:
                color = (0, 255, 0)
            elif traffic_light.state == TrafficLightState.YELLOW:
                color = (0, 255, 255)
            else:
                color = (128, 128, 128)

            cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(output, f"TL: {traffic_light.state.name}",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw person detection
        if self.show_detections and person.detected:
            for (x, y, bw, bh), conf in person.all_detections:
                color = (0, 255, 0) if person.in_pickup_zone else (255, 165, 0)
                cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(output, f"Person: {conf:.2f}",
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw state info panel
        if self.show_state:
            # Background for info panel
            cv2.rectangle(output, (w - 220, 0), (w, 120), (0, 0, 0), -1)
            cv2.rectangle(output, (w - 220, 0), (w, 120), (255, 255, 255), 1)

            # State
            state_color = {
                VehicleState.CRUISE: (0, 255, 0),
                VehicleState.STOP_RED: (0, 0, 255),
                VehicleState.STOP_PERSON: (255, 165, 0),
                VehicleState.PICKUP_WAIT: (255, 255, 0),
            }.get(state, (255, 255, 255))

            cv2.putText(output, f"State: {state.name}",
                        (w - 210, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            # Time in state
            time_in_state = self.state_machine.get_time_in_state()
            cv2.putText(output, f"Time: {time_in_state:.1f}s",
                        (w - 210, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Traffic light
            tl_text = f"TL: {traffic_light.state.name}"
            cv2.putText(output, tl_text,
                        (w - 210, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # FPS
            cv2.putText(output, f"FPS: {self.fps:.1f}",
                        (w - 210, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw steering info
        if self.show_steering:
            # Steering bar
            bar_width = 200
            bar_height = 20
            bar_x = (w - bar_width) // 2
            bar_y = h - 40

            cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                          (50, 50, 50), -1)

            # Center line
            center_x = bar_x + bar_width // 2
            cv2.line(output, (center_x, bar_y), (center_x, bar_y + bar_height),
                     (255, 255, 255), 2)

            # Steering indicator
            steering_pos = int(center_x + lane.steering_suggestion * (bar_width // 2))
            cv2.circle(output, (steering_pos, bar_y + bar_height // 2), 8, (0, 255, 0), -1)

            # Steering text
            cv2.putText(output, f"Steering: {lane.steering_suggestion:.3f}",
                        (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output

    def update_fps(self):
        """Update FPS calculation."""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = current_time

    def run(self):
        """Main processing loop."""
        print("Press 'q' to quit, 's' to toggle state display, 'd' to toggle detections")

        while True:
            # Get frame
            frame = self.camera.get_frame()
            if frame is None:
                print("Warning: Could not get frame")
                time.sleep(0.1)
                continue

            self.frame_count += 1

            # Process frame
            traffic_light, person, lane, state = self.process_frame(frame)

            # Create and send vision message
            msg = self.create_vision_message(traffic_light, person, lane, state)
            self.communicator.send_vision_message(msg)

            # Update FPS
            self.update_fps()

            # Visualize
            if self.show_camera:
                viz = self.visualize(frame, traffic_light, person, lane, state)
                cv2.imshow(self.window_name, viz)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_state = not self.show_state
            elif key == ord('d'):
                self.show_detections = not self.show_detections

        self.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='QCar Vision System - QLabs Camera (NO WEBCAM FALLBACK)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: This system requires QLabs with a spawned QCar.

Before running:
1. Start QLabs (Self-Driving Car Studio)
2. In MATLAB: Run Setup_Competition_Map and Setup_QCar2_Params
3. In MATLAB: Run qcar_main_control (UDP receiver)
4. Then run this script

To verify camera setup first:
    python test_qlabs_camera.py
"""
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration YAML file')
    parser.add_argument('--no-display', action='store_true',
                        help='Run without display window')
    # Keep --webcam for backwards compatibility but make it error
    parser.add_argument('--webcam', action='store_true',
                        help=argparse.SUPPRESS)  # Hidden - deprecated

    args = parser.parse_args()

    # Check for deprecated webcam flag
    if args.webcam:
        print("ERROR: --webcam flag is no longer supported.")
        print("This system requires QLabs with a spawned QCar.")
        print("")
        print("To verify your QLabs setup, run:")
        print("    python test_qlabs_camera.py")
        return 1

    # Create and run vision system (use_webcam=False always)
    try:
        system = VisionSystem(config_path=args.config, use_webcam=False)
    except RuntimeError as e:
        print(f"ERROR: Failed to create vision system: {e}")
        return 1

    if args.no_display:
        system.show_camera = False

    try:
        system.start()
        system.run()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        system.stop()

    return 0


if __name__ == "__main__":
    main()
