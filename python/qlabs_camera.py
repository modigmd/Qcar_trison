"""
QLabs QCar Camera Interface

Provides camera capture from QLabs virtual environment using
Quanser's qvl (QLabs Virtual Library) API.

This is a STRICT implementation - NO webcam fallback.
If QLabs camera fails, Python stops immediately.

Camera IDs (from QLabsQCar2):
- CAMERA_CSI_FRONT: Front CSI camera (820x410)
- CAMERA_CSI_BACK: Back CSI camera
- CAMERA_CSI_LEFT: Left CSI camera
- CAMERA_CSI_RIGHT: Right CSI camera
- CAMERA_RGB: Front RGB camera (640x480)

Author: QCar AV System
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

# Ensure QVL library is on the path (matches Setup_Competition_Map.m)
_qal_dir = os.environ.get('QAL_DIR', '')
if _qal_dir:
    _qvl_path = os.path.join(_qal_dir, '0_libraries', 'python')
    if _qvl_path not in sys.path and os.path.isdir(_qvl_path):
        sys.path.insert(0, _qvl_path)

# qvl imports - these MUST work or we fail
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2


@dataclass
class QLabsCameraConfig:
    """Configuration for QLabs QCar camera."""
    qlabs_address: str = "localhost"
    qlabs_timeout_s: float = 10.0
    qcar_actor_number: int = 0  # MUST match the QCar used by MATLAB
    camera_id: int = QLabsQCar2.CAMERA_CSI_FRONT
    warmup_frames: int = 2  # Drop first couple frames (optional)


class QLabsQCarCamera:
    """
    Strict QLabs camera:
    - MUST connect to QLabs
    - MUST read frames from QCar2.get_image()
    - NO webcam fallback (raises on failure)
    """

    def __init__(self, cfg: QLabsCameraConfig):
        """
        Initialize QLabs QCar camera.

        Args:
            cfg: Camera configuration

        Raises:
            RuntimeError: If QLabs connection fails
        """
        self.cfg = cfg
        self.qlabs = QuanserInteractiveLabs()
        self.qcar: Optional[QLabsQCar2] = None
        self._is_open = False

    def open(self) -> bool:
        """
        Open connection to QLabs and initialize QCar camera.

        Returns:
            True if successful

        Raises:
            RuntimeError: If connection fails
        """
        print(f"Connecting to QLabs at {self.cfg.qlabs_address}...")

        ok = self.qlabs.open(self.cfg.qlabs_address, self.cfg.qlabs_timeout_s)
        if not ok:
            raise RuntimeError(
                f"QLabs open() failed: address={self.cfg.qlabs_address}, "
                f"timeout={self.cfg.qlabs_timeout_s}. "
                "Make sure QLabs is running with Self-Driving Car Studio open."
            )

        print(f"Connected to QLabs successfully")

        # Initialize QCar2 reference
        self.qcar = QLabsQCar2(self.qlabs)
        self.qcar.actorNumber = int(self.cfg.qcar_actor_number)

        print(f"QCar2 initialized with actorNumber={self.qcar.actorNumber}")
        print(f"Using camera_id={self.cfg.camera_id}")

        self._is_open = True

        # Warmup: discard first few frames
        for i in range(max(0, self.cfg.warmup_frames)):
            try:
                _ = self._read_raw()
                print(f"Warmup frame {i+1}/{self.cfg.warmup_frames} OK")
            except RuntimeError as e:
                print(f"Warmup frame {i+1} failed: {e}")

        return True

    def close(self) -> None:
        """Close QLabs connection."""
        self._is_open = False
        try:
            self.qlabs.close()
            print("QLabs connection closed")
        except Exception as e:
            print(f"Warning during QLabs close: {e}")

    def _read_raw(self) -> np.ndarray:
        """
        Read raw frame from QCar camera.

        Returns:
            BGR uint8 image (OpenCV format)

        Raises:
            RuntimeError: If camera read fails
        """
        if self.qcar is None:
            raise RuntimeError("QCar not initialized. Call open() first.")

        status, jpg_bytes = self.qcar.get_image(self.cfg.camera_id)

        if not status or jpg_bytes is None or len(jpg_bytes) == 0:
            raise RuntimeError(
                f"QLabsQCar2.get_image() failed. "
                f"actorNumber={self.qcar.actorNumber}, camera_id={self.cfg.camera_id}. "
                "Make sure the QCar is spawned in QLabs."
            )

        buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise RuntimeError(
                "Failed to decode JPG bytes from QLabs into an image "
                "(cv2.imdecode returned None)."
            )

        return frame_bgr

    def read(self) -> np.ndarray:
        """
        Read a frame from the QLabs QCar camera.

        Returns:
            BGR uint8 image (OpenCV format)

        Raises:
            RuntimeError: If camera read fails
        """
        return self._read_raw()

    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def autodetect_actor_number(
    qlabs_address: str = "localhost",
    timeout_s: float = 10.0,
    camera_id: int = QLabsQCar2.CAMERA_CSI_FRONT,
    max_actor: int = 15,
) -> Tuple[int, Tuple[int, int, int]]:
    """
    Tries actorNumber 0..max_actor and returns the first that yields a decodable frame.

    Args:
        qlabs_address: QLabs server address
        timeout_s: Connection timeout
        camera_id: Camera ID to test
        max_actor: Maximum actor number to try

    Returns:
        Tuple of (actorNumber, frame_shape)

    Raises:
        RuntimeError: If no valid actor found
    """
    print(f"Autodetecting QCar actor number (trying 0..{max_actor})...")

    qlabs = QuanserInteractiveLabs()
    ok = qlabs.open(qlabs_address, timeout_s)
    if not ok:
        raise RuntimeError(
            f"QLabs open() failed: address={qlabs_address}, timeout={timeout_s}"
        )

    qcar = QLabsQCar2(qlabs)
    try:
        for a in range(0, max_actor + 1):
            qcar.actorNumber = a
            status, jpg_bytes = qcar.get_image(camera_id)
            if not status or not jpg_bytes:
                continue
            frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            print(f"Found valid QCar at actorNumber={a}, frame shape={frame.shape}")
            return a, frame.shape
    finally:
        try:
            qlabs.close()
        except Exception:
            pass

    raise RuntimeError(
        f"No actorNumber in 0..{max_actor} returned a valid image for camera_id={camera_id}. "
        "Make sure the QCar is spawned in QLabs."
    )


def get_camera_id_by_name(name: str) -> int:
    """
    Get camera ID constant by name.

    Args:
        name: Camera name ('front', 'back', 'left', 'right', 'rgb')

    Returns:
        Camera ID constant
    """
    camera_map = {
        'front': QLabsQCar2.CAMERA_CSI_FRONT,
        'csi_front': QLabsQCar2.CAMERA_CSI_FRONT,
        'back': QLabsQCar2.CAMERA_CSI_BACK,
        'csi_back': QLabsQCar2.CAMERA_CSI_BACK,
        'left': QLabsQCar2.CAMERA_CSI_LEFT,
        'csi_left': QLabsQCar2.CAMERA_CSI_LEFT,
        'right': QLabsQCar2.CAMERA_CSI_RIGHT,
        'csi_right': QLabsQCar2.CAMERA_CSI_RIGHT,
        'rgb': QLabsQCar2.CAMERA_RGB,
    }
    return camera_map.get(name.lower(), QLabsQCar2.CAMERA_CSI_FRONT)


# =============================================================================
# QCarCameraSystem - Compatibility wrapper for vision_main.py
# =============================================================================

class QCarCameraSystem:
    """
    Camera system wrapper for compatibility with vision_main.py.

    This is a thin wrapper around QLabsQCarCamera that provides
    the interface expected by the existing vision system.

    NO WEBCAM FALLBACK - fails hard if QLabs is not available.
    """

    def __init__(self, config: dict = None, use_mock: bool = False):
        """
        Initialize QCar camera system.

        Args:
            config: Configuration dictionary
            use_mock: IGNORED - webcam fallback is disabled
        """
        if use_mock:
            raise RuntimeError(
                "Webcam/mock mode is disabled. "
                "This system requires QLabs with a spawned QCar. "
                "Remove --webcam flag and ensure QLabs is running."
            )

        self.config = config or {}
        qlabs_config = self.config.get('qlabs', {})

        # Build camera config from yaml
        self.camera_config = QLabsCameraConfig(
            qlabs_address=qlabs_config.get('host', 'localhost'),
            qlabs_timeout_s=qlabs_config.get('timeout', 10.0),
            qcar_actor_number=qlabs_config.get('qcar_actor_number', 0),
            camera_id=get_camera_id_by_name(
                qlabs_config.get('camera', 'front')
            ),
            warmup_frames=qlabs_config.get('warmup_frames', 2),
        )

        self.camera: Optional[QLabsQCarCamera] = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the camera system.

        Returns:
            True if successful

        Raises:
            RuntimeError: If initialization fails
        """
        print("Initializing QLabs QCar camera system (NO FALLBACK MODE)...")

        self.camera = QLabsQCarCamera(self.camera_config)

        try:
            self.camera.open()
            self._initialized = True
            print("QLabs QCar camera initialized successfully")
            return True
        except RuntimeError as e:
            print(f"ERROR: Failed to initialize QLabs camera: {e}")
            print("")
            print("TROUBLESHOOTING:")
            print("1. Is QLabs running with Self-Driving Car Studio open?")
            print("2. Did you run Setup_Competition_Map in MATLAB?")
            print("3. Is the QCar spawned in the virtual environment?")
            print(f"4. Is qcar_actor_number correct? (current: {self.camera_config.qcar_actor_number})")
            print("")
            raise

    def get_frame(self, camera_name: str = 'front') -> Optional[np.ndarray]:
        """
        Get a frame from the camera.

        Args:
            camera_name: Ignored in this implementation (uses configured camera)

        Returns:
            BGR image as numpy array

        Raises:
            RuntimeError: If camera read fails
        """
        if not self._initialized or self.camera is None:
            raise RuntimeError("Camera not initialized. Call initialize() first.")

        return self.camera.read()

    def get_all_frames(self) -> dict:
        """
        Get frames from all cameras (only returns main camera in this implementation).

        Returns:
            Dictionary with 'front' key containing the frame
        """
        frame = self.get_frame()
        return {'front': frame} if frame is not None else {}

    def terminate(self):
        """Terminate the camera system."""
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        self._initialized = False
        print("QLabs QCar camera system terminated")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.terminate()


# =============================================================================
# MAIN - Test the camera
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QLabs QCar Camera Test")
    print("=" * 60)
    print("")

    # Try autodetection first
    try:
        actor_num, shape = autodetect_actor_number(
            camera_id=QLabsQCar2.CAMERA_CSI_FRONT,
            max_actor=15
        )
        print(f"Autodetected: actorNumber={actor_num}, shape={shape}")
    except RuntimeError as e:
        print(f"Autodetection failed: {e}")
        print("Using default actorNumber=0")
        actor_num = 0

    # Test camera
    cfg = QLabsCameraConfig(
        qlabs_address="localhost",
        qcar_actor_number=actor_num,
        camera_id=QLabsQCar2.CAMERA_CSI_FRONT,
    )

    print("")
    print("Testing camera capture...")

    try:
        cam = QLabsQCarCamera(cfg)
        cam.open()

        frame = cam.read()
        print(f"Frame captured: shape={frame.shape}, dtype={frame.dtype}")

        # Save test frame
        cv2.imwrite("qlabs_test_frame.jpg", frame)
        print("Saved: qlabs_test_frame.jpg")

        # Show in window
        print("")
        print("Displaying camera feed. Press 'q' to quit.")
        while True:
            frame = cam.read()
            cv2.imshow("QLabs QCar Camera", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cam.close()
        cv2.destroyAllWindows()

    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("")
        print("Make sure:")
        print("1. QLabs is running (Self-Driving Car Studio)")
        print("2. QCar is spawned (run Setup_Competition_Map in MATLAB)")
        print("3. qcar_actor_number matches your setup")
