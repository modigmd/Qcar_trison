# QCar Autonomous Driving System

Autonomous driving system for the Quanser QCar in the QLabs virtual environment.

---

## Current Progress (Phase 1)

### Completed
- **Architecture Design**: Python + MATLAB UDP communication established
- **Camera Integration**: QLabs camera reading via `qvl` library working
- **Lane Detection**: Yellow line and white curb detection implemented
- **Traffic Light Detection**: HSV color-based detection for red, green, yellow
- **State Machine**: CRUISE, STOP_RED, STOP_PERSON, PICKUP_WAIT states
- **Stanley Controller**: Steering control from path error
- **UDP Communication**: JSON message protocol between Python and MATLAB
- **MATLAB Bridge**: QUARC interface for vehicle actuation

### System Architecture
```
Python (perception + control) --UDP--> MATLAB (QUARC bridge) ---> QLabs (simulation)
```

---

## Future Work (Phase 2)

- **YOLO Person Detection**: YOLOv8 integration for pedestrian detection (currently disabled)
- **Path Planning**: Implement waypoint navigation
- **Obstacle Avoidance**: Dynamic obstacle handling
- **Testing & Validation**: Full system integration testing
- **Performance Optimization**: Frame rate and latency improvements

---

## Quick Start

1. Open QLabs Self-Driving Car Studio
2. Run MATLAB setup scripts (`Setup_Competition_Map`, `Setup_QCar2_Params`)
3. Start MATLAB control: `qcar_main_control`
4. Start Python vision: `python vision_main.py`

---

## Authors

Trisum Team
