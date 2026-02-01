"""
Lane Detection Module using HSV Color Filtering

Based on ODU AV Team approach:
- CSI cameras detect lane lines + curb
- Yellow line on left, white curb on right
- Continuous visual input for lane keeping

Author: QCar AV System
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class LaneDetection:
    """Container for lane detection results."""
    left_line_detected: bool
    right_line_detected: bool
    left_line_points: Optional[List[Tuple[int, int]]]  # Points along left line
    right_line_points: Optional[List[Tuple[int, int]]]  # Points along right line
    lane_center_offset: float  # Offset from lane center (negative = left, positive = right)
    steering_suggestion: float  # Suggested steering value [-1, 1]


class LaneDetector:
    """
    Lane Detector using HSV color filtering.

    Detects:
    - Yellow lane line on the left (solid yellow line)
    - White curb/line on the right

    Uses classical computer vision (no ML).
    """

    def __init__(self, config: dict = None):
        """
        Initialize the lane detector.

        Args:
            config: Configuration dictionary with HSV thresholds
        """
        self.config = config or {}
        lane_config = self.config.get('lane_detection', {})

        # Yellow line thresholds (left lane marker)
        yellow_config = lane_config.get('yellow_line', {})
        self.yellow_lower = np.array(yellow_config.get('lower', [15, 80, 80]))
        self.yellow_upper = np.array(yellow_config.get('upper', [35, 255, 255]))

        # White curb thresholds (right lane marker)
        white_config = lane_config.get('white_curb', {})
        self.white_lower = np.array(white_config.get('lower', [0, 0, 200]))
        self.white_upper = np.array(white_config.get('upper', [180, 30, 255]))

        # ROI (region of interest) as ratios [x, y, width, height]
        self.roi = lane_config.get('roi', [0.0, 0.5, 1.0, 0.5])

        # Morphological kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Lane width in pixels (approximate)
        self.expected_lane_width = 300

        # Steering gain
        self.steering_gain = 0.01

    def _get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract region of interest from frame."""
        h, w = frame.shape[:2]
        x = int(self.roi[0] * w)
        y = int(self.roi[1] * h)
        roi_w = int(self.roi[2] * w)
        roi_h = int(self.roi[3] * h)

        return frame[y:y+roi_h, x:x+roi_w], (x, y)

    def _detect_line(self, hsv: np.ndarray,
                     lower: np.ndarray,
                     upper: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Detect line using color thresholding.

        Args:
            hsv: HSV image
            lower: Lower HSV threshold
            upper: Upper HSV threshold

        Returns:
            Binary mask and list of line points
        """
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return mask, []

        # Find the largest contour (main line)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 100:  # Too small
            return mask, []

        # Get points along the contour
        points = []
        for point in largest:
            points.append(tuple(point[0]))

        # Sort by y-coordinate (bottom to top in image)
        points.sort(key=lambda p: p[1], reverse=True)

        return mask, points

    def _fit_line(self, points: List[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
        """
        Fit a line to points using least squares.

        Args:
            points: List of (x, y) points

        Returns:
            (slope, intercept) or None
        """
        if len(points) < 2:
            return None

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        try:
            # Fit line: x = slope * y + intercept (using y as independent variable)
            coeffs = np.polyfit(y, x, 1)
            return coeffs[0], coeffs[1]
        except:
            return None

    def _calculate_steering(self, left_x: Optional[float],
                           right_x: Optional[float],
                           frame_width: int) -> Tuple[float, float]:
        """
        Calculate steering based on detected lane lines.

        Args:
            left_x: X position of left line at bottom of frame
            right_x: X position of right line at bottom of frame
            frame_width: Width of frame

        Returns:
            (lane_center_offset, steering_suggestion)
        """
        frame_center = frame_width / 2

        if left_x is not None and right_x is not None:
            # Both lines detected - use center of lane
            lane_center = (left_x + right_x) / 2
            offset = frame_center - lane_center
        elif left_x is not None:
            # Only left line - estimate lane center
            lane_center = left_x + self.expected_lane_width / 2
            offset = frame_center - lane_center
        elif right_x is not None:
            # Only right line - estimate lane center
            lane_center = right_x - self.expected_lane_width / 2
            offset = frame_center - lane_center
        else:
            # No lines detected
            return 0.0, 0.0

        # Calculate steering (negative = turn left, positive = turn right)
        steering = np.clip(offset * self.steering_gain, -1.0, 1.0)

        return offset, steering

    def detect(self, frame: np.ndarray) -> LaneDetection:
        """
        Detect lane lines in frame.

        Args:
            frame: Input BGR frame from camera

        Returns:
            LaneDetection with detection results
        """
        if frame is None or frame.size == 0:
            return LaneDetection(
                left_line_detected=False,
                right_line_detected=False,
                left_line_points=None,
                right_line_points=None,
                lane_center_offset=0.0,
                steering_suggestion=0.0
            )

        # Extract ROI
        roi, (x_offset, y_offset) = self._get_roi(frame)
        roi_h, roi_w = roi.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Detect yellow line (left)
        yellow_mask, yellow_points = self._detect_line(hsv, self.yellow_lower, self.yellow_upper)

        # Detect white curb (right)
        white_mask, white_points = self._detect_line(hsv, self.white_lower, self.white_upper)

        # Separate left and right points based on position
        left_points = []
        right_points = []

        # Yellow line should be on the left
        for p in yellow_points:
            if p[0] < roi_w / 2:
                left_points.append((p[0] + x_offset, p[1] + y_offset))

        # White curb should be on the right
        for p in white_points:
            if p[0] > roi_w / 2:
                right_points.append((p[0] + x_offset, p[1] + y_offset))

        # Fit lines and get bottom positions
        left_x = None
        right_x = None

        if left_points:
            line = self._fit_line(left_points)
            if line:
                slope, intercept = line
                left_x = slope * (y_offset + roi_h) + intercept

        if right_points:
            line = self._fit_line(right_points)
            if line:
                slope, intercept = line
                right_x = slope * (y_offset + roi_h) + intercept

        # Calculate steering
        offset, steering = self._calculate_steering(left_x, right_x, frame.shape[1])

        return LaneDetection(
            left_line_detected=len(left_points) > 0,
            right_line_detected=len(right_points) > 0,
            left_line_points=left_points if left_points else None,
            right_line_points=right_points if right_points else None,
            lane_center_offset=offset,
            steering_suggestion=steering
        )

    def visualize(self, frame: np.ndarray, detection: LaneDetection) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Input BGR frame
            detection: Detection result

        Returns:
            Frame with visualization
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        # Draw ROI rectangle
        roi_x = int(self.roi[0] * w)
        roi_y = int(self.roi[1] * h)
        roi_w = int(self.roi[2] * w)
        roi_h = int(self.roi[3] * h)
        cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h),
                      (128, 128, 128), 1)

        # Draw left line points (yellow)
        if detection.left_line_points:
            for i, point in enumerate(detection.left_line_points):
                cv2.circle(output, point, 3, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(output, detection.left_line_points[i-1], point, (0, 255, 255), 2)

        # Draw right line points (white)
        if detection.right_line_points:
            for i, point in enumerate(detection.right_line_points):
                cv2.circle(output, point, 3, (255, 255, 255), -1)
                if i > 0:
                    cv2.line(output, detection.right_line_points[i-1], point, (255, 255, 255), 2)

        # Draw lane center indicator
        center_x = w // 2
        indicator_y = h - 50

        # Draw frame center line
        cv2.line(output, (center_x, h - 100), (center_x, h), (255, 0, 0), 2)

        # Draw steering indicator
        steering_offset = int(detection.steering_suggestion * 100)
        indicator_x = center_x + steering_offset
        cv2.circle(output, (indicator_x, indicator_y), 10, (0, 255, 0), -1)

        # Draw steering text
        steering_text = f"Steering: {detection.steering_suggestion:.3f}"
        cv2.putText(output, steering_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw line detection status
        left_status = "Left: YES" if detection.left_line_detected else "Left: NO"
        right_status = "Right: YES" if detection.right_line_detected else "Right: NO"
        cv2.putText(output, left_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(output, right_status, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output


# Example usage and testing
if __name__ == "__main__":
    # Create detector with default config
    detector = LaneDetector()

    # Test with webcam or sample image
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect lanes
        detection = detector.detect(frame)

        # Visualize
        output = detector.visualize(frame, detection)

        cv2.imshow("Lane Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
