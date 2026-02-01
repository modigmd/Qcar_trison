"""
UDP Communication Module for Python <-> MATLAB IPC

Handles bidirectional UDP communication between:
- Python (sends vision data)
- MATLAB (sends control feedback, receives vision data)

Author: QCar AV System
"""

import socket
import threading
import queue
import time
from typing import Optional, Callable
from dataclasses import dataclass

from .message_protocol import VisionMessage, ControlMessage


@dataclass
class UDPConfig:
    """UDP communication configuration."""
    matlab_host: str = "127.0.0.1"
    matlab_port: int = 5005
    python_host: str = "127.0.0.1"
    python_port: int = 5006
    buffer_size: int = 4096
    timeout: float = 0.1


class UDPCommunicator:
    """
    UDP Communicator for Python <-> MATLAB communication.

    - Sends VisionMessage to MATLAB
    - Receives ControlMessage from MATLAB (optional)
    - Thread-safe operation
    - Non-blocking send/receive
    """

    def __init__(self, config: dict = None):
        """
        Initialize UDP communicator.

        Args:
            config: Configuration dictionary with UDP settings
        """
        self.config = config or {}
        comm_config = self.config.get('communication', {}).get('udp', {})

        self.udp_config = UDPConfig(
            matlab_host=comm_config.get('matlab_host', '127.0.0.1'),
            matlab_port=comm_config.get('matlab_port', 5005),
            python_host=comm_config.get('python_host', '127.0.0.1'),
            python_port=comm_config.get('python_port', 5006),
        )

        # Sockets
        self.send_socket: Optional[socket.socket] = None
        self.recv_socket: Optional[socket.socket] = None

        # Threading
        self.running = False
        self.recv_thread: Optional[threading.Thread] = None
        self.recv_queue: queue.Queue = queue.Queue(maxsize=10)

        # Callbacks
        self.on_control_message: Optional[Callable[[ControlMessage], None]] = None

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.last_send_time = 0.0
        self.last_recv_time = 0.0

    def start(self) -> bool:
        """
        Start the communicator.

        Returns:
            True if successful
        """
        try:
            # Create send socket
            self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.send_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Create receive socket
            self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.recv_socket.settimeout(self.udp_config.timeout)

            # Bind receive socket
            self.recv_socket.bind((
                self.udp_config.python_host,
                self.udp_config.python_port
            ))

            # Start receive thread
            self.running = True
            self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.recv_thread.start()

            print(f"UDP Communicator started:")
            print(f"  Sending to: {self.udp_config.matlab_host}:{self.udp_config.matlab_port}")
            print(f"  Listening on: {self.udp_config.python_host}:{self.udp_config.python_port}")

            return True

        except Exception as e:
            print(f"Error starting UDP communicator: {e}")
            self.stop()
            return False

    def stop(self):
        """Stop the communicator."""
        self.running = False

        if self.recv_thread:
            self.recv_thread.join(timeout=1.0)
            self.recv_thread = None

        if self.send_socket:
            self.send_socket.close()
            self.send_socket = None

        if self.recv_socket:
            self.recv_socket.close()
            self.recv_socket = None

        print("UDP Communicator stopped")

    def send_vision_message(self, message: VisionMessage) -> bool:
        """
        Send a vision message to MATLAB.

        Args:
            message: VisionMessage to send

        Returns:
            True if successful
        """
        if not self.send_socket:
            return False

        try:
            data = message.to_bytes()
            self.send_socket.sendto(
                data,
                (self.udp_config.matlab_host, self.udp_config.matlab_port)
            )
            self.messages_sent += 1
            self.last_send_time = time.time()
            return True

        except Exception as e:
            print(f"Error sending vision message: {e}")
            return False

    def _receive_loop(self):
        """Background thread for receiving messages."""
        while self.running:
            try:
                data, addr = self.recv_socket.recvfrom(self.udp_config.buffer_size)

                if data:
                    try:
                        msg = ControlMessage.from_bytes(data)
                        self.messages_received += 1
                        self.last_recv_time = time.time()

                        # Put in queue (non-blocking)
                        try:
                            self.recv_queue.put_nowait(msg)
                        except queue.Full:
                            # Drop oldest message
                            try:
                                self.recv_queue.get_nowait()
                                self.recv_queue.put_nowait(msg)
                            except:
                                pass

                        # Call callback if registered
                        if self.on_control_message:
                            self.on_control_message(msg)

                    except Exception as e:
                        print(f"Error parsing control message: {e}")

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error in receive loop: {e}")

    def get_control_message(self, timeout: float = 0.0) -> Optional[ControlMessage]:
        """
        Get the latest control message from MATLAB.

        Args:
            timeout: Timeout in seconds (0 = non-blocking)

        Returns:
            ControlMessage or None
        """
        try:
            if timeout > 0:
                return self.recv_queue.get(timeout=timeout)
            else:
                return self.recv_queue.get_nowait()
        except queue.Empty:
            return None

    def get_latest_control_message(self) -> Optional[ControlMessage]:
        """
        Get the most recent control message, discarding older ones.

        Returns:
            Latest ControlMessage or None
        """
        latest = None
        while True:
            try:
                latest = self.recv_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def get_stats(self) -> dict:
        """
        Get communication statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'last_send_time': self.last_send_time,
            'last_recv_time': self.last_recv_time,
            'send_rate': self._calculate_rate(self.messages_sent, self.last_send_time),
            'recv_rate': self._calculate_rate(self.messages_received, self.last_recv_time),
        }

    def _calculate_rate(self, count: int, last_time: float) -> float:
        """Calculate message rate."""
        if last_time == 0:
            return 0.0
        elapsed = time.time() - last_time
        if elapsed > 0:
            return count / elapsed
        return 0.0

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Simple test server for MATLAB simulation
class TestMatlabServer:
    """
    Test server that simulates MATLAB for testing.

    Receives vision messages and sends mock control messages.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.running = False

    def start(self):
        """Start the test server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(0.1)
        self.running = True
        print(f"Test MATLAB server listening on {self.host}:{self.port}")

    def stop(self):
        """Stop the test server."""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None

    def receive_and_respond(self, response_addr: tuple = None) -> Optional[VisionMessage]:
        """
        Receive a vision message and optionally send response.

        Args:
            response_addr: (host, port) to send response, or None

        Returns:
            Received VisionMessage or None
        """
        try:
            data, addr = self.socket.recvfrom(4096)
            msg = VisionMessage.from_bytes(data)
            print(f"Received from {addr}: state={msg.suggested_state}, "
                  f"red={msg.red_light}, green={msg.green_light}")

            # Send mock response if address provided
            if response_addr:
                response = ControlMessage(
                    timestamp=time.time(),
                    current_state=msg.suggested_state,
                    throttle=0.3 if not msg.red_light else 0.0,
                    steering=msg.steering_suggestion,
                    speed=0.3,
                    x=0.0,
                    y=0.0,
                    heading=0.0,
                    last_vision_frame_id=msg.frame_id
                )
                self.socket.sendto(response.to_bytes(), response_addr)

            return msg

        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import time
    from .message_protocol import create_vision_message, VehicleState

    # Test communication
    print("Starting UDP communication test...")

    # Start communicator
    comm = UDPCommunicator()
    comm.start()

    # Start test server in separate thread
    server = TestMatlabServer()
    server.start()

    def server_loop():
        while server.running:
            server.receive_and_respond(("127.0.0.1", 5006))
            time.sleep(0.01)

    server_thread = threading.Thread(target=server_loop, daemon=True)
    server_thread.start()

    # Send test messages
    for i in range(10):
        msg = create_vision_message(
            timestamp=time.time(),
            red_light=(i % 3 == 0),
            green_light=(i % 3 != 0),
            traffic_light_confidence=0.8,
            person_detected=(i % 5 == 0),
            person_in_pickup_zone=(i % 5 == 0),
            lane_detected=True,
            steering_suggestion=0.1 * (i % 3 - 1),
            suggested_state=VehicleState.STOP_RED.value if (i % 3 == 0) else VehicleState.CRUISE.value,
            frame_id=i
        )

        comm.send_vision_message(msg)
        print(f"Sent frame {i}")

        time.sleep(0.1)

        # Check for response
        response = comm.get_latest_control_message()
        if response:
            print(f"  Response: throttle={response.throttle:.2f}, steering={response.steering:.3f}")

    # Cleanup
    time.sleep(0.5)
    server.stop()
    comm.stop()

    print(f"\nStats: {comm.get_stats()}")
