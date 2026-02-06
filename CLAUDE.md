# CLAUDE.md - QCar Autonomous Driving System

## Project Overview

Multi-language autonomous driving system for the Quanser QCar2 in QLabs virtual environment. Python handles perception/vision; MATLAB handles vehicle control via QUARC. Communication between the two is JSON over UDP.

## Repository Structure

```
Qcar_trison/
├── python/                          # Vision & perception system (Python 3.8+)
│   ├── vision_main.py               # Main entry point - orchestrates full pipeline
│   ├── perception/                   # Detection modules
│   │   ├── traffic_light_detector.py # HSV color-based traffic light detection
│   │   ├── lane_detector.py          # Yellow line & white curb detection (HSV)
│   │   └── person_detector.py        # YOLOv8 pedestrian detection
│   ├── communication/                # Inter-process communication
│   │   ├── udp_comm.py               # UDP sender/receiver (threaded)
│   │   └── message_protocol.py       # JSON message format definitions
│   ├── utils/
│   │   └── debug_visualizer.py       # Real-time debug overlay display
│   ├── qlabs_camera.py               # QLabs camera interface (QVL library)
│   ├── qcar_vehicle.py               # Vehicle abstraction (PAL library)
│   ├── setup_qlabs.py                # QLabs environment initialization
│   └── test_qlabs_camera.py          # Camera connectivity verification script
├── matlab/                           # Control layer (MATLAB + QUARC)
│   ├── qcar_main_control.m           # Main control entry point & state machine
│   ├── controllers/
│   │   ├── stanley_controller.m      # Stanley steering algorithm
│   │   └── speed_controller.m        # Throttle/brake regulation
│   └── communication/
│       ├── udp_vision_receiver.m     # UDP message handler
│       ├── udp_read_java.m           # Java DatagramSocket reader
│       └── udp_recv_java.m           # Java DatagramSocket receiver
├── config/
│   └── config.yaml                   # All system parameters (HSV thresholds, gains, ports, etc.)
├── models/                           # Pre-trained ML models (git-ignored, download separately)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview & quick start
```

## Architecture

```
QLabs Simulation
    | (camera frames via QVL)
    v
Python Vision System (vision_main.py)
├── traffic_light_detector  (HSV color filtering)
├── lane_detector           (HSV color filtering)
├── person_detector         (YOLOv8 inference)
├── VisionStateMachine      (CRUISE / STOP_RED / STOP_PERSON / PICKUP_WAIT)
└── UDPCommunicator         (JSON → UDP port 5005)
    |
    v  (JSON messages via UDP)
MATLAB Control System (qcar_main_control.m)
├── UDP receiver            (Java DatagramSocket - no toolbox needed)
├── State machine processor
├── stanley_controller      (steering from lane error)
├── speed_controller        (throttle/brake)
└── QCar actuator           (QUARC interface)
    |
    v
QLabs Virtual Vehicle
```

## Languages & Dependencies

### Python (3.8+)
Install: `pip install -r requirements.txt`

Key dependencies:
- `numpy`, `opencv-python`, `opencv-contrib-python` - core vision
- `ultralytics`, `torch`, `torchvision` - YOLOv8 person detection
- `pyyaml` - config parsing
- `Pillow` - image utilities

External (provided by Quanser, not in pip):
- `pal.utilities` - QCar vehicle control
- `quanser.multimedia` - QLabs integration
- `qvl` - QLabs Virtual Library (camera access)

### MATLAB
- Quanser QUARC required for vehicle actuation
- Uses Java `DatagramSocket` for UDP (no MATLAB toolboxes needed)
- Standard MATLAB only

## Running the System

1. Open QLabs Self-Driving Car Studio
2. Run MATLAB setup: `Setup_Competition_Map`, then `Setup_QCar2_Params`
3. Start MATLAB control: run `qcar_main_control` in MATLAB
4. Start Python vision: `python python/vision_main.py`

Camera verification: `python python/test_qlabs_camera.py`

## Configuration

All tunable parameters live in `config/config.yaml`:
- **QLabs**: host, actor number, camera selection
- **Traffic light**: HSV thresholds for red/green/yellow, blob size limits, ROI
- **Lane detection**: HSV thresholds for yellow line and white curb, ROI
- **Person detection**: YOLO model path, confidence threshold, pickup zone
- **Stanley controller**: gain `k`, softening `epsilon`, max steering angle
- **Speed control**: cruise/turn/approach speeds, braking deceleration
- **State machine**: pickup wait time, green confirm frames, stop distance
- **Communication**: UDP host/port for both Python (5006) and MATLAB (5005)
- **Debug**: visualization toggles, frame saving

## Code Conventions

### Python
- **Dataclasses** for detection results (`TrafficLightDetection`, `PersonDetection`, `LaneDetection`)
- **Factory functions** for creating components (`create_person_detector()`, `create_vehicle()`)
- **Context managers** for camera resources (supports `with` statements)
- **Enum-based state machine** states: `CRUISE`, `STOP_RED`, `STOP_PERSON`, `PICKUP_WAIT`
- **Temporal filtering** with detection history (5-frame voting for stability)
- **HSV color space** for traffic light and lane detection (not deep learning)
- **Thread-safe UDP** using `queue.Queue` for received messages
- Each module has `if __name__ == "__main__"` blocks for standalone testing
- Package-level `__init__.py` files export public APIs

### MATLAB
- Helper/utility functions defined at the end of script files
- Configuration passed as structs with default handling
- State constants defined at script top (`CRUISE=0`, `STOP_RED=1`, etc.)
- Zero-toolbox design - uses only core MATLAB + Java for networking

### Communication Protocol
- JSON messages over UDP (human-readable, debuggable)
- Python sends `VisionMessage` to MATLAB (port 5005): traffic light state, lane error, person detection, vehicle state
- MATLAB sends `ControlMessage` to Python (port 5006): speed, steering, state acknowledgment
- Non-blocking sockets with timeouts to prevent hangs

## Testing

No automated test suite. Testing is done via:
- `python/test_qlabs_camera.py` - verifies QLabs camera connectivity and captures a test frame
- Individual module `__main__` blocks for standalone verification
- Mock implementations available: `MockQCarVehicle`, `PersonDetector` fallback mode, `TestMatlabServer`

## Key Files to Know

| File | Lines | Purpose |
|------|-------|---------|
| `python/vision_main.py` | ~626 | Main orchestrator - camera loop, detection pipeline, state machine, UDP send |
| `matlab/qcar_main_control.m` | ~443 | Main control loop - receives vision data, runs controllers, actuates vehicle |
| `config/config.yaml` | 189 | All system parameters |
| `python/perception/traffic_light_detector.py` | ~350 | HSV-based traffic light color detection |
| `python/perception/lane_detector.py` | ~300 | Yellow/white lane boundary detection |
| `python/perception/person_detector.py` | ~340 | YOLOv8 pedestrian detection wrapper |
| `python/communication/udp_comm.py` | ~310 | Threaded UDP communication layer |
| `python/utils/debug_visualizer.py` | ~440 | Real-time debug overlay rendering |
| `matlab/controllers/stanley_controller.m` | ~140 | Stanley lateral steering algorithm |

## Development Notes

- **No CI/CD pipeline** - all testing is manual against QLabs
- **No build system** - Python runs directly, MATLAB runs in MATLAB IDE
- **Model files are git-ignored** - YOLO weights (`*.pt`, `*.onnx`) must be downloaded separately into `models/`
- **Phase 2 items** (not yet implemented): waypoint path planning, dynamic obstacle avoidance, full integration tests, performance optimization
- **QCar actor number** may vary - use `test_qlabs_camera.py` to auto-detect (tries 0-15)
- HSV thresholds in config may need tuning if QLabs lighting conditions change
