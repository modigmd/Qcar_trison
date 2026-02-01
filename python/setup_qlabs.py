"""
QLabs Setup Script for QCar

This script initializes the QLabs environment and spawns the QCar.
Must be run BEFORE starting the vision system.

Based on Quanser's Setup_Competition_Map.m but in Python.

Prerequisites:
- QLabs must be running (Self-Driving Car Studio > Plane)
- Quanser QVL library must be installed

Author: QCar AV System
"""

import sys
import time
import numpy as np

# Try to import QVL library
try:
    from qvl.qlabs import QuanserInteractiveLabs
    from qvl.qcar import QLabsQCar
    from qvl.free_camera import QLabsFreeCamera
    from qvl.traffic_light import QLabsTrafficLight
    from qvl.crosswalk import QLabsCrosswalk
    from qvl.stop_sign import QLabsStopSign
    from qvl.yield_sign import QLabsYieldSign
    from qvl.roundabout_sign import QLabsRoundaboutSign
    from qvl.person import QLabsPerson
    QVL_AVAILABLE = True
except ImportError as e:
    print(f"QVL library not available: {e}")
    print("This script requires the Quanser QVL library.")
    print("Please ensure Quanser software is properly installed.")
    QVL_AVAILABLE = False


# ============================================================================
# SPAWN LOCATIONS
# These match the Quanser competition map locations
# ============================================================================

SPAWN_LOCATIONS = {
    1: {
        'position': [10.235, -3.355, 0.005],
        'rotation': [0, 0, 180],  # degrees
        'description': 'Start position near parking area'
    },
    2: {
        'position': [-7.35, 3.36, 0.005],
        'rotation': [0, 0, 90],
        'description': 'Alternative start position'
    },
    3: {
        'position': [0, 0, 0.005],
        'rotation': [0, 0, 0],
        'description': 'Center of map'
    }
}


def setup_qlabs_environment(spawn_location: int = 1,
                            qcar_id: int = 0,
                            spawn_traffic: bool = True):
    """
    Set up QLabs environment with QCar and optional traffic elements.

    Args:
        spawn_location: Spawn location ID (1, 2, or 3)
        qcar_id: QCar actor ID
        spawn_traffic: Whether to spawn traffic lights, signs, etc.

    Returns:
        tuple: (qlabs, qcar) objects or (None, None) if failed
    """
    if not QVL_AVAILABLE:
        print("ERROR: QVL library not available")
        return None, None

    print("=" * 60)
    print("QLabs Setup Script for QCar")
    print("=" * 60)

    # Connect to QLabs
    print("\n1. Connecting to QLabs...")
    qlabs = QuanserInteractiveLabs()

    try:
        qlabs.open("localhost")
        print("   Connected to QLabs successfully")
    except Exception as e:
        print(f"   ERROR: Failed to connect to QLabs: {e}")
        print("   Make sure QLabs is running with Self-Driving Car Studio > Plane")
        return None, None

    # Destroy any existing actors (clean slate)
    print("\n2. Cleaning up existing actors...")
    try:
        qlabs.destroy_all_spawned_actors()
        time.sleep(0.5)
        print("   Cleared existing actors")
    except Exception as e:
        print(f"   Warning: Could not clear actors: {e}")

    # Get spawn location
    if spawn_location not in SPAWN_LOCATIONS:
        print(f"   Warning: Invalid spawn location {spawn_location}, using 1")
        spawn_location = 1

    spawn_config = SPAWN_LOCATIONS[spawn_location]
    position = spawn_config['position']
    rotation = spawn_config['rotation']

    print(f"\n3. Spawning QCar at location {spawn_location}...")
    print(f"   Position: {position}")
    print(f"   Rotation: {rotation}")
    print(f"   Description: {spawn_config['description']}")

    # Spawn QCar
    try:
        qcar = QLabsQCar(qlabs)
        qcar.spawn_id(
            actorNumber=qcar_id,
            location=position,
            rotation=rotation,
            scale=[1, 1, 1],
            configuration=0,
            waitForConfirmation=True
        )
        print("   QCar spawned successfully")
    except Exception as e:
        print(f"   ERROR: Failed to spawn QCar: {e}")
        return qlabs, None

    # Set up camera view
    print("\n4. Setting up camera view...")
    try:
        # Possess the QCar camera (third-person view)
        qcar.possess(qcar.CAMERA_TRAILING)
        print("   Camera set to trailing view")
    except Exception as e:
        print(f"   Warning: Could not set camera: {e}")

    # Spawn traffic elements if requested
    if spawn_traffic:
        print("\n5. Spawning traffic elements...")
        spawn_traffic_elements(qlabs)

    print("\n" + "=" * 60)
    print("Setup complete! QCar is ready.")
    print("=" * 60)
    print(f"\nQCar ID: {qcar_id}")
    print("You can now run the vision system.")

    return qlabs, qcar


def spawn_traffic_elements(qlabs):
    """Spawn traffic lights, signs, and pedestrians."""

    # Traffic Light positions (approximate - adjust for your map)
    traffic_light_positions = [
        {'pos': [5.55, 0.87, 0.0], 'rot': [0, 0, 0]},
        {'pos': [-5.55, -0.87, 0.0], 'rot': [0, 0, 180]},
    ]

    # Spawn traffic lights
    for i, tl in enumerate(traffic_light_positions):
        try:
            light = QLabsTrafficLight(qlabs)
            light.spawn_id(
                actorNumber=i,
                location=tl['pos'],
                rotation=tl['rot'],
                scale=[1, 1, 1],
                configuration=0,
                waitForConfirmation=True
            )
            # Start with red light
            light.set_state(state=QLabsTrafficLight.STATE_RED)
            print(f"   Spawned traffic light {i}")
        except Exception as e:
            print(f"   Warning: Could not spawn traffic light {i}: {e}")

    # Spawn stop signs
    stop_sign_positions = [
        {'pos': [8.0, 2.0, 0.0], 'rot': [0, 0, 90]},
    ]

    for i, ss in enumerate(stop_sign_positions):
        try:
            sign = QLabsStopSign(qlabs)
            sign.spawn_id(
                actorNumber=i,
                location=ss['pos'],
                rotation=ss['rot'],
                scale=[1, 1, 1],
                configuration=0,
                waitForConfirmation=True
            )
            print(f"   Spawned stop sign {i}")
        except Exception as e:
            print(f"   Warning: Could not spawn stop sign {i}: {e}")

    # Spawn a pedestrian for pickup testing
    try:
        person = QLabsPerson(qlabs)
        person.spawn_id(
            actorNumber=0,
            location=[6.0, -2.0, 0.0],
            rotation=[0, 0, 0],
            scale=[1, 1, 1],
            configuration=0,  # Random appearance
            waitForConfirmation=True
        )
        print("   Spawned pedestrian for pickup testing")
    except Exception as e:
        print(f"   Warning: Could not spawn pedestrian: {e}")


def traffic_light_controller(qlabs, cycle_time: float = 30.0):
    """
    Simple traffic light controller loop.

    Run this in a separate thread or process to control traffic lights.

    Args:
        qlabs: QLabs connection
        cycle_time: Full cycle time in seconds
    """
    print("\nStarting traffic light controller...")
    print("Press Ctrl+C to stop")

    try:
        light0 = QLabsTrafficLight(qlabs, actorNumber=0)
        light1 = QLabsTrafficLight(qlabs, actorNumber=1)

        while True:
            # Light 0 green, Light 1 red
            light0.set_state(QLabsTrafficLight.STATE_GREEN)
            light1.set_state(QLabsTrafficLight.STATE_RED)
            time.sleep(cycle_time / 2 - 3)

            # Yellow transition
            light0.set_state(QLabsTrafficLight.STATE_YELLOW)
            time.sleep(3)

            # Light 0 red, Light 1 green
            light0.set_state(QLabsTrafficLight.STATE_RED)
            light1.set_state(QLabsTrafficLight.STATE_GREEN)
            time.sleep(cycle_time / 2 - 3)

            # Yellow transition
            light1.set_state(QLabsTrafficLight.STATE_YELLOW)
            time.sleep(3)

    except KeyboardInterrupt:
        print("\nTraffic light controller stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='QLabs Setup for QCar')
    parser.add_argument('--spawn', type=int, default=1, choices=[1, 2, 3],
                        help='Spawn location (1, 2, or 3)')
    parser.add_argument('--qcar-id', type=int, default=0,
                        help='QCar actor ID')
    parser.add_argument('--no-traffic', action='store_true',
                        help='Do not spawn traffic elements')
    parser.add_argument('--run-traffic-controller', action='store_true',
                        help='Run traffic light controller after setup')

    args = parser.parse_args()

    if not QVL_AVAILABLE:
        print("\nERROR: This script requires the Quanser QVL library.")
        print("Please ensure Quanser software is properly installed.")
        print("\nAlternatively, use MATLAB to set up the environment:")
        print("  1. Open QLabs")
        print("  2. Run Setup_Competition_Map.m in MATLAB")
        sys.exit(1)

    # Set up environment
    qlabs, qcar = setup_qlabs_environment(
        spawn_location=args.spawn,
        qcar_id=args.qcar_id,
        spawn_traffic=not args.no_traffic
    )

    if qlabs is None or qcar is None:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)

    # Optionally run traffic controller
    if args.run_traffic_controller:
        traffic_light_controller(qlabs)
    else:
        print("\nSetup complete. You can now run:")
        print("  python vision_main.py --config ../config/config.yaml")
        print("\nOr run the traffic controller in another terminal:")
        print("  python setup_qlabs.py --run-traffic-controller")

        # Keep connection alive
        print("\nPress Ctrl+C to close QLabs connection...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    # Cleanup
    print("\nClosing QLabs connection...")
    qlabs.close()


if __name__ == "__main__":
    main()
