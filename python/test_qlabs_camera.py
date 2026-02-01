#!/usr/bin/env python3
"""
QLabs QCar Camera Test Script

Quick verification that Python can read frames from the QLabs QCar camera.

Usage:
    1. Start QLabs (Self-Driving Car Studio)
    2. Run Setup_Competition_Map in MATLAB to spawn the QCar
    3. Run this script:
       python test_qlabs_camera.py

If successful, this will:
    - Autodetect the QCar actor number
    - Capture a test frame
    - Save it as qlabs_first_frame.jpg
    - Display the frame shape

Author: QCar AV System
"""

import sys
import cv2

try:
    from qvl.qcar2 import QLabsQCar2
    from qlabs_camera import (
        QLabsCameraConfig,
        QLabsQCarCamera,
        autodetect_actor_number
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("")
    print("Make sure:")
    print("1. PYTHONPATH includes Quanser_Academic_Resources/0_libraries/python")
    print("2. qvl package is installed")
    sys.exit(1)


def main():
    print("=" * 60)
    print("QLabs QCar Camera Test")
    print("=" * 60)
    print("")

    # Step 1: Autodetect actor number
    print("Step 1: Autodetecting QCar actor number...")
    print("-" * 40)

    try:
        actor_num, shape = autodetect_actor_number(
            qlabs_address="localhost",
            camera_id=QLabsQCar2.CAMERA_CSI_FRONT,
            max_actor=15
        )
        print(f"SUCCESS: Found QCar at actorNumber={actor_num}")
        print(f"Frame shape: {shape}")
    except RuntimeError as e:
        print(f"FAILED: {e}")
        print("")
        print("Troubleshooting:")
        print("1. Is QLabs running?")
        print("2. Is Self-Driving Car Studio open?")
        print("3. Did you run Setup_Competition_Map in MATLAB?")
        sys.exit(1)

    print("")

    # Step 2: Test camera capture
    print("Step 2: Testing camera capture...")
    print("-" * 40)

    cfg = QLabsCameraConfig(
        qlabs_address="localhost",
        qcar_actor_number=actor_num,
        camera_id=QLabsQCar2.CAMERA_CSI_FRONT,
        warmup_frames=0,  # No warmup for test
    )

    try:
        cam = QLabsQCarCamera(cfg)
        cam.open()

        frame = cam.read()
        print(f"SUCCESS: Frame captured")
        print(f"  Shape: {frame.shape}")
        print(f"  Dtype: {frame.dtype}")
        print(f"  Min/Max: {frame.min()}/{frame.max()}")

        # Save frame
        output_file = "qlabs_first_frame.jpg"
        cv2.imwrite(output_file, frame)
        print(f"  Saved: {output_file}")

        cam.close()

    except RuntimeError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print("")
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("")
    print(f"Add this to your config.yaml:")
    print(f"")
    print(f"qlabs:")
    print(f"  qcar_actor_number: {actor_num}")
    print("")
    print("Now you can run: python vision_main.py --config ../config/config.yaml")


if __name__ == "__main__":
    main()
