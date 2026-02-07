# CLAUDE.md - QCar Autonomous Driving System

## Project Overview

Multi-language autonomous driving system for the Quanser QCar2 in QLabs virtual environment. Python handles perception/vision; MATLAB handles vehicle control via QUARC. Communication between the two is JSON over UDP.

**Competition context**: Quanser ACC Self-Driving Car Competition. The official competition stack uses MATLAB/Simulink + QUARC. Our system adds a Python vision layer communicating with the MATLAB control layer over UDP.

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
│   ├── setup_qlabs.py                # QLabs environment initialization (QCar2)
│   └── test_qlabs_camera.py          # Camera connectivity verification script
├── matlab/                           # Control layer (MATLAB + QUARC)
│   ├── qcar_main_control.m           # Main control entry point & state machine
│   ├── controllers/
│   │   ├── stanley_controller.m      # Stanley steering algorithm
│   │   └── speed_controller.m        # Throttle/brake regulation
│   └── communication/
│       ├── udp_vision_receiver.m     # UDP message handler (MATLAB udpport)
│       ├── udp_recv_java.m           # Java DatagramSocket receiver (no toolbox)
│       ├── udp_read_java.m           # Java DatagramSocket reader (no toolbox)
│       └── udp_send_java.m           # Java DatagramSocket sender (no toolbox)
├── config/
│   └── config.yaml                   # All system parameters (HSV thresholds, gains, ports, etc.)
├── models/                           # Pre-trained ML models (git-ignored, download separately)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview & quick start
```

## Architecture

```
QLabs Simulation
    | (camera frames via QVL: qvl.qcar2.QLabsQCar2.get_image())
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
├── UDP sender              (Java DatagramSocket → port 5006 feedback)
└── QCar actuator           (QUARC interface / mock for testing)
    |
    v
QLabs Virtual Vehicle
```

## Camera Pipeline (QCar → YOLO/Detection)

The camera pipeline works as follows:
1. `qlabs_camera.py` connects to QLabs via `qvl.qlabs.QuanserInteractiveLabs.open("localhost")`
2. Creates `qvl.qcar2.QLabsQCar2` reference with the correct `actorNumber`
3. Calls `QLabsQCar2.get_image(camera_id)` which returns `(status, jpg_bytes)`
4. Decodes JPEG bytes: `cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)` → BGR numpy array
5. This BGR frame is fed directly to all detectors (traffic light HSV, lane HSV, YOLO person)

**Important**: The QVL library path must be available. Set `QAL_DIR` environment variable to your Quanser Academic Resources directory. The code auto-adds `$QAL_DIR/0_libraries/python` to `sys.path`.

## Languages & Dependencies

### Python (3.8+)
Install: `pip install -r requirements.txt`

Key dependencies:
- `numpy`, `opencv-python`, `opencv-contrib-python` - core vision
- `ultralytics`, `torch`, `torchvision` - YOLOv8 person detection
- `pyyaml` - config parsing
- `Pillow` - image utilities

External (provided by Quanser, not in pip):
- `qvl` - QLabs Virtual Library (camera access via `QLabsQCar2.get_image()`)
- `pal.utilities` - QCar vehicle control (optional, for direct Python control)

### MATLAB
- Quanser QUARC required for vehicle actuation (or use Simulink stack)
- Uses Java `DatagramSocket` for UDP (no MATLAB toolboxes needed)
- Standard MATLAB only
- Official competition Simulink model: `VIRTUAL_self_driving_stack_v2.slx`

## Running the System

### Prerequisites
1. Set environment variable: `QAL_DIR` pointing to Quanser Academic Resources
2. Open QLabs and select Self-Driving Car Studio > Plane

### Option A: MATLAB setup (recommended, matches competition)
1. In MATLAB: `Setup_Competition_Map` (spawns QCar2 and environment)
2. In MATLAB: `Setup_QCar2_Params`
3. Start MATLAB control: `qcar_main_control` or run Simulink model
4. Start Python vision: `python python/vision_main.py`

### Option B: Python setup
1. `python python/setup_qlabs.py` (spawns QCar2 via QVL API)
2. Start MATLAB control: `qcar_main_control`
3. Start Python vision: `python python/vision_main.py`

### Camera verification
```
python python/test_qlabs_camera.py
```

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
- Zero-toolbox design - uses Java `DatagramSocket` for UDP networking
- `udp_recv_java.m` / `udp_read_java.m` / `udp_send_java.m` for UDP I/O

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

| File | Purpose |
|------|---------|
| `python/vision_main.py` | Main orchestrator - camera loop, detection pipeline, state machine, UDP send |
| `matlab/qcar_main_control.m` | Main control loop - receives vision data, runs controllers, actuates vehicle |
| `config/config.yaml` | All system parameters |
| `python/qlabs_camera.py` | QVL camera interface (QLabsQCar2.get_image → OpenCV BGR) |
| `python/perception/traffic_light_detector.py` | HSV-based traffic light color detection |
| `python/perception/lane_detector.py` | Yellow/white lane boundary detection |
| `python/perception/person_detector.py` | YOLOv8 pedestrian detection wrapper |
| `python/communication/udp_comm.py` | Threaded UDP communication layer |
| `python/setup_qlabs.py` | Python-based QLabs environment setup (QCar2) |
| `matlab/controllers/stanley_controller.m` | Stanley lateral steering algorithm |
| `matlab/communication/udp_send_java.m` | Java-based UDP sender for MATLAB feedback |

## Development Notes

- **No CI/CD pipeline** - all testing is manual against QLabs
- **No build system** - Python runs directly, MATLAB runs in MATLAB IDE
- **Model files are git-ignored** - YOLO weights (`*.pt`, `*.onnx`) must be downloaded into `models/`. Person detector auto-searches `models/` directory and falls back to ultralytics auto-download
- **QCar2 scale is 1/10** in the virtual competition (matches `Setup_Competition_Map.m`)
- **QCar actor number** may vary - use `test_qlabs_camera.py` to auto-detect (tries 0-15)
- **QAL_DIR env var** must be set for QVL imports. Points to Quanser Academic Resources root
- HSV thresholds in config may need tuning if QLabs lighting conditions change
- **Phase 2 items** (not yet implemented): waypoint path planning, dynamic obstacle avoidance, full integration tests, performance optimization
- **Official competition stack** is Simulink-based (`VIRTUAL_self_driving_stack_v2.slx`). This Python+MATLAB architecture is a custom extension for vision processing
