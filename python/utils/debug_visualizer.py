"""
Debug Visualization Module for QCar Vision System

Provides comprehensive visual debugging with:
- Multi-camera view
- Detection overlays
- State information
- Steering/throttle indicators
- Real-time plots

Based on ODU AV Team demo visualization style.

Author: QCar AV System
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from collections import deque
import time


@dataclass
class DebugConfig:
    """Configuration for debug visualization."""
    window_width: int = 1280
    window_height: int = 720
    show_fps: bool = True
    show_state: bool = True
    show_detections: bool = True
    show_steering_bar: bool = True
    show_mini_plot: bool = True
    plot_history: int = 100
    font_scale: float = 0.6
    line_thickness: int = 2


class DebugVisualizer:
    """
    Comprehensive debug visualizer for QCar vision system.

    Creates a unified debug view combining:
    - Main camera feed with detections
    - State information panel
    - Steering indicator
    - Mini real-time plots
    """

    # Color scheme
    COLORS = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'gray': (128, 128, 128),
        'orange': (0, 165, 255),
        'panel_bg': (30, 30, 30),
        'panel_border': (80, 80, 80),
    }

    # State colors
    STATE_COLORS = {
        0: (0, 255, 0),      # CRUISE - green
        1: (0, 0, 255),      # STOP_RED - red
        2: (0, 165, 255),    # STOP_PERSON - orange
        3: (0, 255, 255),    # PICKUP_WAIT - yellow
        4: (255, 0, 255),    # STOP_SIGN - magenta
    }

    STATE_NAMES = {
        0: 'CRUISE',
        1: 'STOP_RED',
        2: 'STOP_PERSON',
        3: 'PICKUP_WAIT',
        4: 'STOP_SIGN',
    }

    def __init__(self, config: DebugConfig = None):
        """
        Initialize the debug visualizer.

        Args:
            config: Debug configuration
        """
        self.config = config or DebugConfig()

        # History for plots
        self.steering_history = deque(maxlen=self.config.plot_history)
        self.speed_history = deque(maxlen=self.config.plot_history)
        self.cte_history = deque(maxlen=self.config.plot_history)

        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        # Window
        self.window_name = "QCar Vision Debug"

    def update_fps(self):
        """Update FPS calculation."""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time

        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

    def create_debug_view(self,
                          frame: np.ndarray,
                          state: int,
                          steering: float,
                          speed: float = 0.0,
                          throttle: float = 0.0,
                          detections: Dict[str, Any] = None,
                          extra_info: Dict[str, Any] = None) -> np.ndarray:
        """
        Create comprehensive debug visualization.

        Args:
            frame: Main camera frame (BGR)
            state: Current vehicle state (0-4)
            steering: Steering angle [-1, 1] or radians
            speed: Current speed (m/s)
            throttle: Throttle command [0, 1]
            detections: Dict with detection results
            extra_info: Additional info to display

        Returns:
            Debug visualization frame
        """
        self.update_fps()

        # Add to history
        self.steering_history.append(steering)
        self.speed_history.append(speed)

        # Create canvas
        canvas = np.zeros((self.config.window_height, self.config.window_width, 3), dtype=np.uint8)
        canvas[:] = self.COLORS['panel_bg']

        # Layout:
        # +------------------------+-------------+
        # |                        |   State     |
        # |     Main Camera        |   Panel     |
        # |                        |             |
        # +------------------------+-------------+
        # |     Steering Bar       |   Mini Plot |
        # +------------------------+-------------+

        main_width = int(self.config.window_width * 0.7)
        main_height = int(self.config.window_height * 0.75)
        panel_width = self.config.window_width - main_width
        bottom_height = self.config.window_height - main_height

        # Draw main camera view
        self._draw_main_camera(canvas, frame, main_width, main_height, detections)

        # Draw state panel
        self._draw_state_panel(canvas, main_width, panel_width, main_height,
                               state, speed, throttle, extra_info)

        # Draw steering bar
        self._draw_steering_bar(canvas, main_width, main_height, bottom_height, steering)

        # Draw mini plot
        if self.config.show_mini_plot:
            self._draw_mini_plot(canvas, main_width, main_height, panel_width, bottom_height)

        # Draw FPS
        if self.config.show_fps:
            cv2.putText(canvas, f"FPS: {self.current_fps:.1f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.COLORS['white'], 1)

        return canvas

    def _draw_main_camera(self, canvas: np.ndarray, frame: np.ndarray,
                          width: int, height: int, detections: Dict[str, Any]):
        """Draw main camera view with detections."""
        if frame is None:
            # Draw placeholder
            cv2.rectangle(canvas, (0, 0), (width, height), self.COLORS['gray'], 2)
            cv2.putText(canvas, "No Camera Feed", (width // 2 - 80, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLORS['white'], 2)
            return

        # Resize frame to fit
        frame_resized = cv2.resize(frame, (width, height))

        # Copy to canvas
        canvas[0:height, 0:width] = frame_resized

        # Draw border
        cv2.rectangle(canvas, (0, 0), (width - 1, height - 1),
                      self.COLORS['panel_border'], 2)

    def _draw_state_panel(self, canvas: np.ndarray, x_start: int, width: int,
                          height: int, state: int, speed: float, throttle: float,
                          extra_info: Dict[str, Any]):
        """Draw state information panel."""
        # Panel background
        cv2.rectangle(canvas, (x_start, 0), (x_start + width, height),
                      self.COLORS['panel_bg'], -1)
        cv2.rectangle(canvas, (x_start, 0), (x_start + width, height),
                      self.COLORS['panel_border'], 2)

        # Title
        cv2.putText(canvas, "VEHICLE STATUS",
                    (x_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, self.COLORS['cyan'], 2)

        # Horizontal separator
        cv2.line(canvas, (x_start + 10, 45), (x_start + width - 10, 45),
                 self.COLORS['panel_border'], 1)

        y_pos = 75

        # State
        state_color = self.STATE_COLORS.get(state, self.COLORS['white'])
        state_name = self.STATE_NAMES.get(state, 'UNKNOWN')
        cv2.putText(canvas, "State:", (x_start + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['white'], 1)
        cv2.putText(canvas, state_name, (x_start + 10, y_pos + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

        y_pos += 70

        # Speed
        cv2.putText(canvas, f"Speed: {speed:.2f} m/s",
                    (x_start + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, self.COLORS['white'], 1)

        y_pos += 30

        # Throttle bar
        cv2.putText(canvas, "Throttle:",
                    (x_start + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.COLORS['gray'], 1)

        bar_x = x_start + 80
        bar_width = width - 100
        bar_height = 15
        cv2.rectangle(canvas, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3),
                      self.COLORS['gray'], 1)
        fill_width = int(bar_width * min(throttle, 1.0))
        if fill_width > 0:
            cv2.rectangle(canvas, (bar_x, y_pos - 12), (bar_x + fill_width, y_pos + 3),
                          self.COLORS['green'], -1)

        y_pos += 40

        # Detection status section
        cv2.line(canvas, (x_start + 10, y_pos - 10), (x_start + width - 10, y_pos - 10),
                 self.COLORS['panel_border'], 1)
        cv2.putText(canvas, "DETECTIONS",
                    (x_start + 10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, self.COLORS['cyan'], 1)

        y_pos += 40

        # Traffic light status
        tl_color = self.COLORS['gray']
        tl_text = "None"
        if extra_info:
            if extra_info.get('red_light'):
                tl_color = self.COLORS['red']
                tl_text = "RED"
            elif extra_info.get('green_light'):
                tl_color = self.COLORS['green']
                tl_text = "GREEN"
            elif extra_info.get('yellow_light'):
                tl_color = self.COLORS['yellow']
                tl_text = "YELLOW"

        cv2.circle(canvas, (x_start + 25, y_pos), 10, tl_color, -1)
        cv2.putText(canvas, f"Traffic: {tl_text}",
                    (x_start + 45, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.COLORS['white'], 1)

        y_pos += 35

        # Person detection
        person_detected = extra_info.get('person_detected', False) if extra_info else False
        in_zone = extra_info.get('person_in_pickup_zone', False) if extra_info else False

        person_color = self.COLORS['gray']
        person_text = "None"
        if person_detected:
            if in_zone:
                person_color = self.COLORS['orange']
                person_text = "IN ZONE!"
            else:
                person_color = self.COLORS['yellow']
                person_text = "Detected"

        cv2.circle(canvas, (x_start + 25, y_pos), 10, person_color, -1)
        cv2.putText(canvas, f"Person: {person_text}",
                    (x_start + 45, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.COLORS['white'], 1)

        y_pos += 35

        # Lane detection
        lane_detected = extra_info.get('lane_detected', False) if extra_info else False
        lane_color = self.COLORS['green'] if lane_detected else self.COLORS['gray']
        lane_text = "OK" if lane_detected else "None"

        cv2.circle(canvas, (x_start + 25, y_pos), 10, lane_color, -1)
        cv2.putText(canvas, f"Lane: {lane_text}",
                    (x_start + 45, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.COLORS['white'], 1)

    def _draw_steering_bar(self, canvas: np.ndarray, width: int, y_start: int,
                           height: int, steering: float):
        """Draw steering indicator bar."""
        # Background
        cv2.rectangle(canvas, (0, y_start), (width, y_start + height),
                      self.COLORS['panel_bg'], -1)
        cv2.rectangle(canvas, (0, y_start), (width - 1, y_start + height - 1),
                      self.COLORS['panel_border'], 2)

        # Title
        cv2.putText(canvas, f"Steering: {steering:.3f}",
                    (10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, self.COLORS['white'], 1)

        # Bar
        bar_margin = 50
        bar_width = width - 2 * bar_margin
        bar_height = 30
        bar_x = bar_margin
        bar_y = y_start + 40

        # Bar background
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      self.COLORS['panel_border'], 1)

        # Center line
        center_x = bar_x + bar_width // 2
        cv2.line(canvas, (center_x, bar_y), (center_x, bar_y + bar_height),
                 self.COLORS['white'], 2)

        # Tick marks
        for i in range(-4, 5):
            tick_x = center_x + int(i * bar_width / 10)
            cv2.line(canvas, (tick_x, bar_y + bar_height - 5),
                     (tick_x, bar_y + bar_height), self.COLORS['gray'], 1)

        # Indicator
        # Clamp steering to [-1, 1]
        steering_clamped = max(-1, min(1, steering))
        indicator_x = int(center_x + steering_clamped * (bar_width // 2))

        # Indicator color based on direction
        if abs(steering_clamped) < 0.05:
            indicator_color = self.COLORS['green']
        elif steering_clamped < 0:
            indicator_color = self.COLORS['cyan']  # Left
        else:
            indicator_color = self.COLORS['magenta']  # Right

        cv2.circle(canvas, (indicator_x, bar_y + bar_height // 2), 12,
                   indicator_color, -1)
        cv2.circle(canvas, (indicator_x, bar_y + bar_height // 2), 12,
                   self.COLORS['white'], 2)

        # Labels
        cv2.putText(canvas, "L", (bar_x - 15, bar_y + bar_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['cyan'], 1)
        cv2.putText(canvas, "R", (bar_x + bar_width + 5, bar_y + bar_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['magenta'], 1)

    def _draw_mini_plot(self, canvas: np.ndarray, x_start: int, y_start: int,
                        width: int, height: int):
        """Draw mini real-time plot."""
        # Background
        cv2.rectangle(canvas, (x_start, y_start), (x_start + width, y_start + height),
                      self.COLORS['panel_bg'], -1)
        cv2.rectangle(canvas, (x_start, y_start), (x_start + width - 1, y_start + height - 1),
                      self.COLORS['panel_border'], 2)

        # Title
        cv2.putText(canvas, "Steering History",
                    (x_start + 10, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, self.COLORS['cyan'], 1)

        # Plot area
        plot_margin = 20
        plot_x = x_start + plot_margin
        plot_y = y_start + 30
        plot_width = width - 2 * plot_margin
        plot_height = height - 50

        # Plot background
        cv2.rectangle(canvas, (plot_x, plot_y),
                      (plot_x + plot_width, plot_y + plot_height),
                      (20, 20, 20), -1)

        # Center line (zero steering)
        center_y = plot_y + plot_height // 2
        cv2.line(canvas, (plot_x, center_y), (plot_x + plot_width, center_y),
                 self.COLORS['gray'], 1)

        # Plot steering history
        if len(self.steering_history) > 1:
            points = []
            for i, s in enumerate(self.steering_history):
                x = plot_x + int(i * plot_width / self.config.plot_history)
                y = int(center_y - s * (plot_height // 2))
                y = max(plot_y, min(plot_y + plot_height, y))
                points.append((x, y))

            # Draw line
            for i in range(1, len(points)):
                cv2.line(canvas, points[i-1], points[i], self.COLORS['green'], 1)

    def show(self, frame: np.ndarray):
        """Display frame in window."""
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        """Wait for key press."""
        return cv2.waitKey(delay) & 0xFF

    def close(self):
        """Close all windows."""
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    import random

    visualizer = DebugVisualizer()

    # Simulate frames
    for i in range(500):
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Simulate state changes
        state = (i // 100) % 4
        steering = 0.3 * np.sin(i * 0.05) + random.uniform(-0.02, 0.02)
        speed = 0.3 * (1 if state == 0 else 0)
        throttle = 0.5 if state == 0 else 0.0

        extra_info = {
            'red_light': state == 1,
            'green_light': state == 0,
            'yellow_light': False,
            'person_detected': state == 2 or state == 3,
            'person_in_pickup_zone': state == 2 or state == 3,
            'lane_detected': True,
        }

        # Create debug view
        debug_frame = visualizer.create_debug_view(
            frame, state, steering, speed, throttle,
            detections=None, extra_info=extra_info
        )

        # Show
        visualizer.show(debug_frame)

        key = visualizer.wait_key(30)
        if key == ord('q'):
            break

    visualizer.close()
