"""
QCar Vehicle Interface for QLabs

Provides vehicle control interface using Quanser's PAL library.
Wraps the QCar hardware abstraction for throttle, steering, and LED control.

Based on York-SDCNLab ACC-2025-Phase1 implementation.

Author: QCar AV System
"""

import numpy as np
import time
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# VEHICLE CONFIGURATION
# ============================================================================

@dataclass
class VehicleConfig:
    """Configuration for QCar vehicle."""
    # Control coefficients
    throttle_coefficient: float = 0.3   # Max throttle scaling
    steering_coefficient: float = 0.5   # Max steering scaling

    # Physical parameters
    wheelbase: float = 0.256            # Wheelbase in meters (for QCar)
    max_speed: float = 1.0              # Max speed in m/s

    # LED configuration
    num_leds: int = 8


class LEDState(Enum):
    """LED indicator states."""
    OFF = 0
    GREEN = 1       # Normal driving
    RED = 2         # Stopping (traffic light, stop sign)
    BLUE = 3        # Pickup/dropoff
    YELLOW = 4      # Caution


class QCarVehicle:
    """
    QCar Vehicle control interface.

    Provides methods for:
    - Throttle and steering control
    - Speed estimation from wheel encoders
    - LED indicator control
    - Vehicle state management
    """

    def __init__(self, config: VehicleConfig = None):
        """
        Initialize QCar vehicle interface.

        Args:
            config: Vehicle configuration
        """
        self.config = config or VehicleConfig()

        # PAL availability
        self._pal_available = False
        self._qcar = None

        # State
        self.is_initialized = False
        self.current_throttle = 0.0
        self.current_steering = 0.0
        self.current_speed = 0.0

        # LED state
        self._led_state = np.zeros(self.config.num_leds, dtype=np.float32)

    def initialize(self) -> bool:
        """
        Initialize the vehicle interface.

        Returns:
            True if successful
        """
        try:
            from pal.products.qcar import QCar

            self._pal_available = True

            # Initialize QCar with control coefficients
            self._qcar = QCar(
                readMode=1,  # 0=non-blocking, 1=blocking
                frequency=500  # Control frequency in Hz
            )

            self.is_initialized = True
            print("QCar vehicle initialized")
            return True

        except ImportError as e:
            print(f"PAL library not available: {e}")
            print("QCar vehicle control will be simulated")
            self.is_initialized = True  # Allow mock operation
            return True
        except Exception as e:
            print(f"Error initializing QCar: {e}")
            return False

    def set_control(self, throttle: float, steering: float) -> bool:
        """
        Set vehicle throttle and steering.

        Args:
            throttle: Throttle command [-1, 1] (negative = reverse)
            steering: Steering command [-1, 1] (negative = left, positive = right)

        Returns:
            True if successful
        """
        # Apply coefficients and clamp
        throttle_cmd = np.clip(
            throttle * self.config.throttle_coefficient,
            -self.config.throttle_coefficient,
            self.config.throttle_coefficient
        )
        steering_cmd = np.clip(
            steering * self.config.steering_coefficient,
            -self.config.steering_coefficient,
            self.config.steering_coefficient
        )

        self.current_throttle = throttle_cmd
        self.current_steering = steering_cmd

        if self._qcar is not None:
            try:
                # Write to QCar hardware
                self._qcar.read_write_std(
                    throttle=throttle_cmd,
                    steering=steering_cmd,
                    LEDs=self._led_state
                )
                return True
            except Exception as e:
                print(f"Error setting control: {e}")
                return False
        else:
            # Mock mode - just store values
            return True

    def halt(self) -> bool:
        """
        Stop the vehicle (zero throttle).

        Returns:
            True if successful
        """
        return self.set_control(0.0, self.current_steering)

    def emergency_stop(self) -> bool:
        """
        Emergency stop - zero throttle and steering.

        Returns:
            True if successful
        """
        self._led_state[:] = 0.5  # Flash all LEDs
        return self.set_control(0.0, 0.0)

    def estimate_speed(self) -> float:
        """
        Estimate vehicle speed from wheel encoders.

        Returns:
            Estimated speed in m/s
        """
        if self._qcar is not None:
            try:
                # Read from motor encoders
                # The QCar's read_write_std returns encoder data
                motor_speed = self._qcar.motorTach if hasattr(self._qcar, 'motorTach') else 0.0

                # Convert encoder reading to speed
                # This is a simplified conversion - actual conversion depends on gearing
                wheel_radius = 0.033  # QCar wheel radius in meters
                gear_ratio = 1.0

                self.current_speed = abs(motor_speed) * wheel_radius / gear_ratio
                return self.current_speed
            except Exception:
                return self.current_speed
        else:
            # Mock mode - estimate from throttle
            # Simple simulation: speed approaches throttle * max_speed
            target_speed = abs(self.current_throttle) * self.config.max_speed
            # Exponential approach to target
            alpha = 0.1
            self.current_speed = alpha * target_speed + (1 - alpha) * self.current_speed
            return self.current_speed

    def set_leds(self, state: LEDState):
        """
        Set LED indicator state.

        Args:
            state: LED state (OFF, GREEN, RED, BLUE, YELLOW)
        """
        if state == LEDState.OFF:
            self._led_state[:] = 0.0
        elif state == LEDState.GREEN:
            # Front and rear green
            self._led_state[:] = 0.0
            self._led_state[0] = 1.0  # Front left
            self._led_state[1] = 1.0  # Front right
        elif state == LEDState.RED:
            # Brake lights (rear)
            self._led_state[:] = 0.0
            self._led_state[6] = 1.0  # Rear left
            self._led_state[7] = 1.0  # Rear right
        elif state == LEDState.BLUE:
            # All blue (pickup mode)
            self._led_state[:] = 0.0
            self._led_state[2] = 1.0  # Side indicators
            self._led_state[3] = 1.0
        elif state == LEDState.YELLOW:
            # Hazard/caution
            self._led_state[:] = 1.0

    def set_turn_signal(self, direction: str):
        """
        Set turn signal.

        Args:
            direction: 'left', 'right', or 'off'
        """
        if direction == 'left':
            self._led_state[0] = 1.0
            self._led_state[2] = 1.0
            self._led_state[4] = 1.0
            self._led_state[6] = 1.0
        elif direction == 'right':
            self._led_state[1] = 1.0
            self._led_state[3] = 1.0
            self._led_state[5] = 1.0
            self._led_state[7] = 1.0
        else:
            # Auto-determine from steering
            if self.current_steering < -0.3:
                self.set_turn_signal('left')
            elif self.current_steering > 0.3:
                self.set_turn_signal('right')

    def get_state(self) -> dict:
        """
        Get current vehicle state.

        Returns:
            Dictionary with vehicle state
        """
        return {
            'throttle': self.current_throttle,
            'steering': self.current_steering,
            'speed': self.current_speed,
            'is_initialized': self.is_initialized,
            'pal_available': self._pal_available,
        }

    def terminate(self):
        """Terminate the vehicle interface."""
        # Stop vehicle
        self.emergency_stop()

        if self._qcar is not None:
            try:
                self._qcar.terminate()
            except:
                pass
            self._qcar = None

        self.is_initialized = False
        print("QCar vehicle terminated")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.terminate()


class MockQCarVehicle(QCarVehicle):
    """
    Mock QCar vehicle for testing without hardware.

    Simulates vehicle dynamics for testing perception and control.
    """

    def __init__(self, config: VehicleConfig = None):
        """Initialize mock vehicle."""
        super().__init__(config)

        # Simulated position and heading
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0

        # Simulation time step
        self.dt = 0.033  # ~30 Hz

    def initialize(self) -> bool:
        """Initialize mock vehicle."""
        self.is_initialized = True
        print("Mock QCar vehicle initialized")
        return True

    def set_control(self, throttle: float, steering: float) -> bool:
        """Set control and simulate motion."""
        # Store commands
        self.current_throttle = np.clip(throttle, -1.0, 1.0)
        self.current_steering = np.clip(steering, -1.0, 1.0)

        # Simulate vehicle dynamics (bicycle model)
        self._simulate_step()

        return True

    def _simulate_step(self):
        """Simulate one time step of vehicle motion."""
        # Update speed
        target_speed = self.current_throttle * self.config.max_speed
        alpha = 0.3
        self.current_speed = alpha * target_speed + (1 - alpha) * self.current_speed

        # Update position (bicycle model)
        if abs(self.current_speed) > 0.001:
            # Steering angle to wheel angle
            wheel_angle = self.current_steering * 0.5  # Max 0.5 rad

            # Update heading
            angular_velocity = (self.current_speed / self.config.wheelbase) * np.tan(wheel_angle)
            self.heading += angular_velocity * self.dt

            # Normalize heading
            while self.heading > np.pi:
                self.heading -= 2 * np.pi
            while self.heading < -np.pi:
                self.heading += 2 * np.pi

            # Update position
            self.x += self.current_speed * np.cos(self.heading) * self.dt
            self.y += self.current_speed * np.sin(self.heading) * self.dt

    def get_pose(self) -> Tuple[float, float, float]:
        """
        Get simulated vehicle pose.

        Returns:
            (x, y, heading) in meters and radians
        """
        return self.x, self.y, self.heading

    def set_pose(self, x: float, y: float, heading: float):
        """Set simulated vehicle pose."""
        self.x = x
        self.y = y
        self.heading = heading

    def get_state(self) -> dict:
        """Get vehicle state including pose."""
        state = super().get_state()
        state['x'] = self.x
        state['y'] = self.y
        state['heading'] = self.heading
        return state

    def terminate(self):
        """Terminate mock vehicle."""
        self.is_initialized = False
        print("Mock QCar vehicle terminated")


def create_vehicle(config: VehicleConfig = None, use_mock: bool = False) -> QCarVehicle:
    """
    Factory function to create QCar vehicle.

    Args:
        config: Vehicle configuration
        use_mock: Force mock vehicle

    Returns:
        QCarVehicle instance
    """
    if use_mock:
        return MockQCarVehicle(config)

    # Try real vehicle first
    try:
        from pal.products.qcar import QCar
        return QCarVehicle(config)
    except ImportError:
        print("PAL library not available, using mock vehicle")
        return MockQCarVehicle(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("QCar Vehicle Test")
    print("=" * 50)

    # Test with mock vehicle
    vehicle = create_vehicle(use_mock=True)

    if vehicle.initialize():
        print("\nRunning vehicle test...")

        # Test control
        for i in range(100):
            # Simple figure-8 pattern
            t = i * 0.1
            throttle = 0.3
            steering = 0.5 * np.sin(t * 0.5)

            vehicle.set_control(throttle, steering)

            state = vehicle.get_state()
            print(f"Step {i}: speed={state['speed']:.2f}, "
                  f"x={state.get('x', 0):.2f}, y={state.get('y', 0):.2f}")

            time.sleep(0.033)

        vehicle.terminate()
    else:
        print("Failed to initialize vehicle")
