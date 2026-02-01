"""
Person Detection Module using YOLOv8

Based on ODU AV Team Stage 2 approach:
- Uses YOLO-based neural network for real-time object detection
- Detects 'person' class for pedestrian pickup scenarios
- No training required - uses pretrained COCO weights

Author: QCar AV System
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

# Try to import ultralytics, fall back to mock if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Person detection will be disabled.")
    print("Install with: pip install ultralytics")


@dataclass
class PersonDetection:
    """Container for person detection results."""
    detected: bool
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    center: Optional[Tuple[int, int]]  # (cx, cy)
    in_pickup_zone: bool
    all_detections: List[Tuple[Tuple[int, int, int, int], float]]  # List of (box, confidence)


class PersonDetector:
    """
    Person Detector using YOLOv8.

    Uses a pretrained YOLOv8 model (COCO weights) to detect people.
    Specifically designed for pedestrian pickup scenarios in autonomous vehicles.
    """

    # COCO class ID for 'person'
    PERSON_CLASS_ID = 0

    def __init__(self, config: dict = None):
        """
        Initialize the person detector.

        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or {}
        person_config = self.config.get('person_detection', {})

        # Model configuration
        self.model_name = person_config.get('model', 'yolov8n.pt')
        self.confidence_threshold = person_config.get('confidence_threshold', 0.5)
        self.classes = person_config.get('classes', [self.PERSON_CLASS_ID])

        # Pickup zone configuration (relative to frame)
        pickup_zone = person_config.get('pickup_zone', {})
        self.pickup_zone = {
            'x_min': pickup_zone.get('x_min', 0.0),
            'x_max': pickup_zone.get('x_max', 0.4),
            'y_min': pickup_zone.get('y_min', 0.3),
            'y_max': pickup_zone.get('y_max', 0.9)
        }

        # Load model
        self.model = None
        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.model_name)
            print(f"Loaded YOLO model: {self.model_name}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def _is_in_pickup_zone(self, box: Tuple[int, int, int, int],
                           frame_shape: Tuple[int, int]) -> bool:
        """
        Check if detection is in the pickup zone.

        Args:
            box: Bounding box (x, y, w, h)
            frame_shape: Frame dimensions (height, width)

        Returns:
            True if the detection center is in the pickup zone
        """
        h, w = frame_shape[:2]
        x, y, bw, bh = box

        # Calculate center of bounding box
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h

        # Check if center is within pickup zone
        in_zone = (self.pickup_zone['x_min'] <= cx <= self.pickup_zone['x_max'] and
                   self.pickup_zone['y_min'] <= cy <= self.pickup_zone['y_max'])

        return in_zone

    def detect(self, frame: np.ndarray) -> PersonDetection:
        """
        Detect people in frame.

        Args:
            frame: Input BGR frame from camera

        Returns:
            PersonDetection with detection results
        """
        if frame is None or frame.size == 0:
            return PersonDetection(
                detected=False,
                confidence=0.0,
                bounding_box=None,
                center=None,
                in_pickup_zone=False,
                all_detections=[]
            )

        if self.model is None:
            # Return mock detection if model not available
            return PersonDetection(
                detected=False,
                confidence=0.0,
                bounding_box=None,
                center=None,
                in_pickup_zone=False,
                all_detections=[]
            )

        # Run YOLO inference
        results = self.model(frame, verbose=False, classes=self.classes,
                             conf=self.confidence_threshold)

        # Process results
        all_detections = []
        best_detection = None
        best_confidence = 0.0
        person_in_pickup_zone = False

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                # Get bounding box coordinates (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                w = x2 - x1
                h = y2 - y1

                # Get confidence
                conf = float(box.conf[0])

                # Get class
                cls = int(box.cls[0])

                # Only process person class
                if cls != self.PERSON_CLASS_ID:
                    continue

                detection_box = (x1, y1, w, h)
                all_detections.append((detection_box, conf))

                # Check if in pickup zone
                in_zone = self._is_in_pickup_zone(detection_box, frame.shape)
                if in_zone:
                    person_in_pickup_zone = True

                    # Track the best detection in pickup zone
                    if conf > best_confidence:
                        best_confidence = conf
                        best_detection = detection_box

        # If no detection in pickup zone, use highest confidence overall
        if best_detection is None and all_detections:
            best_detection, best_confidence = max(all_detections, key=lambda x: x[1])

        # Create result
        if best_detection:
            x, y, w, h = best_detection
            center = (x + w // 2, y + h // 2)

            return PersonDetection(
                detected=True,
                confidence=best_confidence,
                bounding_box=best_detection,
                center=center,
                in_pickup_zone=person_in_pickup_zone,
                all_detections=all_detections
            )
        else:
            return PersonDetection(
                detected=False,
                confidence=0.0,
                bounding_box=None,
                center=None,
                in_pickup_zone=False,
                all_detections=[]
            )

    def visualize(self, frame: np.ndarray, detection: PersonDetection) -> np.ndarray:
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

        # Draw pickup zone
        zone_x1 = int(self.pickup_zone['x_min'] * w)
        zone_y1 = int(self.pickup_zone['y_min'] * h)
        zone_x2 = int(self.pickup_zone['x_max'] * w)
        zone_y2 = int(self.pickup_zone['y_max'] * h)

        zone_color = (0, 255, 255) if detection.in_pickup_zone else (128, 128, 128)
        cv2.rectangle(output, (zone_x1, zone_y1), (zone_x2, zone_y2), zone_color, 2)
        cv2.putText(output, "Pickup Zone", (zone_x1, zone_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1)

        # Draw all detections
        for (x, y, bw, bh), conf in detection.all_detections:
            # Determine if this detection is in pickup zone
            in_zone = self._is_in_pickup_zone((x, y, bw, bh), frame.shape)

            if in_zone:
                color = (0, 255, 0)  # Green for in-zone
            else:
                color = (255, 165, 0)  # Orange for out-of-zone

            cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)

            # Draw label
            label = f"Person ({conf:.2f})"
            cv2.putText(output, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw status
        if detection.in_pickup_zone:
            status_text = "PERSON IN PICKUP ZONE!"
            status_color = (0, 0, 255)
        elif detection.detected:
            status_text = "Person detected (not in zone)"
            status_color = (255, 165, 0)
        else:
            status_text = "No person detected"
            status_color = (128, 128, 128)

        cv2.putText(output, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return output


class MockPersonDetector(PersonDetector):
    """
    Mock person detector for testing without YOLO.

    Simulates person detection based on color detection (for testing).
    """

    def __init__(self, config: dict = None):
        """Initialize mock detector."""
        self.config = config or {}
        person_config = self.config.get('person_detection', {})

        self.confidence_threshold = person_config.get('confidence_threshold', 0.5)

        pickup_zone = person_config.get('pickup_zone', {})
        self.pickup_zone = {
            'x_min': pickup_zone.get('x_min', 0.0),
            'x_max': pickup_zone.get('x_max', 0.4),
            'y_min': pickup_zone.get('y_min', 0.3),
            'y_max': pickup_zone.get('y_max', 0.9)
        }

        self.model = None  # No model needed for mock

    def detect(self, frame: np.ndarray) -> PersonDetection:
        """
        Mock detection using simple color detection.

        This is a placeholder for testing the system without YOLO.
        In the real system, this would be replaced with actual YOLO inference.
        """
        if frame is None or frame.size == 0:
            return PersonDetection(
                detected=False,
                confidence=0.0,
                bounding_box=None,
                center=None,
                in_pickup_zone=False,
                all_detections=[]
            )

        # Simple skin tone detection as placeholder
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Skin tone range (very approximate)
        lower = np.array([0, 20, 70])
        upper = np.array([20, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return PersonDetection(
                detected=False,
                confidence=0.0,
                bounding_box=None,
                center=None,
                in_pickup_zone=False,
                all_detections=[]
            )

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 500:  # Too small
            return PersonDetection(
                detected=False,
                confidence=0.0,
                bounding_box=None,
                center=None,
                in_pickup_zone=False,
                all_detections=[]
            )

        x, y, w, h = cv2.boundingRect(largest)
        confidence = min(area / 5000, 1.0)

        in_zone = self._is_in_pickup_zone((x, y, w, h), frame.shape)

        return PersonDetection(
            detected=True,
            confidence=confidence,
            bounding_box=(x, y, w, h),
            center=(x + w // 2, y + h // 2),
            in_pickup_zone=in_zone,
            all_detections=[((x, y, w, h), confidence)]
        )


def create_detector(config: dict = None, use_mock: bool = False) -> PersonDetector:
    """
    Factory function to create appropriate person detector.

    Args:
        config: Configuration dictionary
        use_mock: Force use of mock detector

    Returns:
        PersonDetector instance
    """
    if use_mock or not YOLO_AVAILABLE:
        print("Using mock person detector")
        return MockPersonDetector(config)
    else:
        return PersonDetector(config)


# Example usage and testing
if __name__ == "__main__":
    # Create detector
    detector = create_detector()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        detection = detector.detect(frame)

        # Visualize
        output = detector.visualize(frame, detection)

        cv2.imshow("Person Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
