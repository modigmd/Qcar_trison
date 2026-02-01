"""
Traffic Light Detection Module using HSV Color Filtering

Based on ODU AV Team approach:
- Front camera detects traffic lights using HSV color filtering
- Detects RED, GREEN, and YELLOW lights
- Returns bounding boxes and detection status

Author: QCar AV System
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class TrafficLightState(Enum):
    """Traffic light state enumeration."""
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


@dataclass
class TrafficLightDetection:
    """Container for traffic light detection results."""
    state: TrafficLightState
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    center: Optional[Tuple[int, int]]  # (cx, cy)


class TrafficLightDetector:
    """
    Traffic Light Detector using HSV color filtering.

    Uses classical computer vision (no ML) to detect traffic lights:
    1. Convert RGB to HSV color space
    2. Apply color thresholds for red/green/yellow
    3. Clean masks using morphological operations
    4. Find contours and bounding boxes
    """

    def __init__(self, config: dict = None):
        """
        Initialize the traffic light detector.

        Args:
            config: Configuration dictionary with HSV thresholds
        """
        # Default HSV thresholds (can be overridden by config)
        self.config = config or {}

        # Red light thresholds (red wraps around in HSV, so we need two ranges)
        tl_config = self.config.get('traffic_light', {})
        red_config = tl_config.get('red', {})
        self.red_lower1 = np.array(red_config.get('lower', [0, 100, 100]))
        self.red_upper1 = np.array(red_config.get('upper', [10, 255, 255]))
        self.red_lower2 = np.array(red_config.get('lower2', [160, 100, 100]))
        self.red_upper2 = np.array(red_config.get('upper2', [180, 255, 255]))

        # Green light thresholds
        green_config = tl_config.get('green', {})
        self.green_lower = np.array(green_config.get('lower', [40, 50, 50]))
        self.green_upper = np.array(green_config.get('upper', [90, 255, 255]))

        # Yellow light thresholds
        yellow_config = tl_config.get('yellow', {})
        self.yellow_lower = np.array(yellow_config.get('lower', [15, 100, 100]))
        self.yellow_upper = np.array(yellow_config.get('upper', [35, 255, 255]))

        # Detection parameters
        self.min_area = tl_config.get('min_area', 100)
        self.max_area = tl_config.get('max_area', 10000)
        self.aspect_ratio_range = tl_config.get('aspect_ratio_range', [0.5, 2.0])

        # ROI (region of interest) as ratios [x, y, width, height]
        self.roi = tl_config.get('roi', [0.3, 0.0, 0.4, 0.5])

        # Morphological kernel for noise removal
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # State tracking for temporal filtering
        self.detection_history = []
        self.history_length = 5

    def _get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract region of interest from frame.

        Args:
            frame: Input BGR frame

        Returns:
            ROI image and (x_offset, y_offset)
        """
        h, w = frame.shape[:2]
        x = int(self.roi[0] * w)
        y = int(self.roi[1] * h)
        roi_w = int(self.roi[2] * w)
        roi_h = int(self.roi[3] * h)

        return frame[y:y+roi_h, x:x+roi_w], (x, y)

    def _detect_color(self, hsv: np.ndarray,
                      lower: np.ndarray,
                      upper: np.ndarray,
                      lower2: np.ndarray = None,
                      upper2: np.ndarray = None) -> np.ndarray:
        """
        Create a binary mask for a specific color range.

        Args:
            hsv: HSV image
            lower: Lower HSV threshold
            upper: Upper HSV threshold
            lower2: Optional second lower threshold (for red)
            upper2: Optional second upper threshold (for red)

        Returns:
            Binary mask
        """
        mask = cv2.inRange(hsv, lower, upper)

        # Handle red which wraps around in HSV
        if lower2 is not None and upper2 is not None:
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

        # Morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask

    def _find_light_contours(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find contours that could be traffic lights.

        Args:
            mask: Binary mask

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (traffic lights are roughly circular/square)
            aspect_ratio = w / h if h > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue

            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Traffic lights should be somewhat circular
                if circularity < 0.3:
                    continue

            valid_boxes.append((x, y, w, h, area))

        # Sort by area (largest first) and return boxes without area
        valid_boxes.sort(key=lambda b: b[4], reverse=True)
        return [(x, y, w, h) for x, y, w, h, _ in valid_boxes]

    def detect(self, frame: np.ndarray) -> TrafficLightDetection:
        """
        Detect traffic light in frame.

        Args:
            frame: Input BGR frame from camera

        Returns:
            TrafficLightDetection with state and bounding box
        """
        if frame is None or frame.size == 0:
            return TrafficLightDetection(
                state=TrafficLightState.UNKNOWN,
                confidence=0.0,
                bounding_box=None,
                center=None
            )

        # Extract ROI
        roi, (x_offset, y_offset) = self._get_roi(frame)

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Detect each color
        red_mask = self._detect_color(hsv, self.red_lower1, self.red_upper1,
                                       self.red_lower2, self.red_upper2)
        green_mask = self._detect_color(hsv, self.green_lower, self.green_upper)
        yellow_mask = self._detect_color(hsv, self.yellow_lower, self.yellow_upper)

        # Find contours for each color
        red_boxes = self._find_light_contours(red_mask)
        green_boxes = self._find_light_contours(green_mask)
        yellow_boxes = self._find_light_contours(yellow_mask)

        # Determine the most likely traffic light state
        best_detection = None
        best_confidence = 0.0

        # Check red lights
        if red_boxes:
            box = red_boxes[0]  # Take the largest
            confidence = self._calculate_confidence(red_mask, box)
            if confidence > best_confidence:
                best_confidence = confidence
                best_detection = (TrafficLightState.RED, box)

        # Check green lights
        if green_boxes:
            box = green_boxes[0]
            confidence = self._calculate_confidence(green_mask, box)
            if confidence > best_confidence:
                best_confidence = confidence
                best_detection = (TrafficLightState.GREEN, box)

        # Check yellow lights
        if yellow_boxes:
            box = yellow_boxes[0]
            confidence = self._calculate_confidence(yellow_mask, box)
            if confidence > best_confidence:
                best_confidence = confidence
                best_detection = (TrafficLightState.YELLOW, box)

        # Create result
        if best_detection:
            state, (x, y, w, h) = best_detection
            # Adjust coordinates back to full frame
            x += x_offset
            y += y_offset
            center = (x + w // 2, y + h // 2)

            return TrafficLightDetection(
                state=state,
                confidence=best_confidence,
                bounding_box=(x, y, w, h),
                center=center
            )
        else:
            return TrafficLightDetection(
                state=TrafficLightState.UNKNOWN,
                confidence=0.0,
                bounding_box=None,
                center=None
            )

    def _calculate_confidence(self, mask: np.ndarray, box: Tuple[int, int, int, int]) -> float:
        """
        Calculate confidence score for a detection.

        Args:
            mask: Binary mask
            box: Bounding box (x, y, w, h)

        Returns:
            Confidence score [0, 1]
        """
        x, y, w, h = box
        roi_mask = mask[y:y+h, x:x+w]

        if roi_mask.size == 0:
            return 0.0

        # Calculate fill ratio within bounding box
        fill_ratio = np.sum(roi_mask > 0) / roi_mask.size

        # Higher fill ratio = higher confidence
        return min(fill_ratio * 1.5, 1.0)

    def detect_with_temporal_filter(self, frame: np.ndarray) -> TrafficLightDetection:
        """
        Detect traffic light with temporal filtering for stability.

        Args:
            frame: Input BGR frame

        Returns:
            Temporally filtered detection
        """
        detection = self.detect(frame)

        # Add to history
        self.detection_history.append(detection.state)
        if len(self.detection_history) > self.history_length:
            self.detection_history.pop(0)

        # Count votes for each state
        state_counts = {}
        for state in self.detection_history:
            state_counts[state] = state_counts.get(state, 0) + 1

        # Find most common state
        most_common = max(state_counts, key=state_counts.get)

        # Return detection with temporally filtered state
        return TrafficLightDetection(
            state=most_common,
            confidence=detection.confidence,
            bounding_box=detection.bounding_box,
            center=detection.center
        )

    def visualize(self, frame: np.ndarray, detection: TrafficLightDetection) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Input BGR frame
            detection: Detection result

        Returns:
            Frame with visualization
        """
        output = frame.copy()

        # Draw ROI rectangle
        h, w = frame.shape[:2]
        roi_x = int(self.roi[0] * w)
        roi_y = int(self.roi[1] * h)
        roi_w = int(self.roi[2] * w)
        roi_h = int(self.roi[3] * h)
        cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h),
                      (128, 128, 128), 1)

        if detection.bounding_box:
            x, y, w, h = detection.bounding_box

            # Choose color based on state
            if detection.state == TrafficLightState.RED:
                color = (0, 0, 255)  # Red in BGR
                label = "RED"
            elif detection.state == TrafficLightState.GREEN:
                color = (0, 255, 0)  # Green in BGR
                label = "GREEN"
            elif detection.state == TrafficLightState.YELLOW:
                color = (0, 255, 255)  # Yellow in BGR
                label = "YELLOW"
            else:
                color = (255, 255, 255)
                label = "UNKNOWN"

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label_text = f"{label} ({detection.confidence:.2f})"
            cv2.putText(output, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output


# Example usage and testing
if __name__ == "__main__":
    # Create detector with default config
    detector = TrafficLightDetector()

    # Test with a sample image or webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect traffic light
        detection = detector.detect_with_temporal_filter(frame)

        # Visualize
        output = detector.visualize(frame, detection)

        # Display state
        state_text = f"State: {detection.state.name}"
        cv2.putText(output, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Traffic Light Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
