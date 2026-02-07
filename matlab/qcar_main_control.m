% QCAR_MAIN_CONTROL - Main control script for QCar autonomous driving
%
% This script integrates:
%   - UDP communication with Python vision system
%   - Stanley steering controller
%   - Speed controller with state-based throttle override
%   - State machine for traffic lights and pedestrian stops
%
% Based on ODU AV Team architecture:
%   - Python handles perception (traffic lights, person detection, lanes)
%   - MATLAB handles control (steering, throttle, state machine)
%
% Prerequisites:
%   - Quanser QUARC installed
%   - QLabs running with QCar spawned
%   - Python vision_main.py running
%
% Author: QCar AV System

%% Clear and setup
clc;
clear;
close all;

% Add paths
addpath('controllers');
addpath('communication');
addpath('models');

%% Configuration
config = struct();

% Vehicle selection (1 = QCar1, 2 = QCar2)
config.qcar_id = 2;

% Control parameters
config.stanley = struct();
config.stanley.k = 2.5;              % Cross-track error gain
config.stanley.epsilon = 0.1;        % Softening constant
config.stanley.max_steering = 0.5;   % Max steering (rad)

config.speed = struct();
config.speed.cruise_speed = 0.3;     % Normal driving speed (m/s)
config.speed.turn_speed = 0.2;       % Speed at turns
config.speed.approach_speed = 0.25;  % Speed approaching intersections
config.speed.Kp = 2.0;               % Speed controller gain
config.speed.max_throttle = 0.5;     % Max throttle

% State machine
config.state = struct();
config.state.pickup_wait_time = 3.0; % Seconds to wait at pickup

% UDP communication
config.udp = struct();
config.udp.local_port = 5005;        % MATLAB listens on this port
config.udp.remote_port = 5006;       % Python listens on this port

% Simulation
config.sim = struct();
config.sim.dt = 0.033;               % Control loop period (~30 Hz)
config.sim.max_time = 300;           % Maximum runtime (seconds)

%% Vehicle State Enumeration
% Matches Python VehicleState enum
CRUISE = 0;
STOP_RED = 1;
STOP_PERSON = 2;
PICKUP_WAIT = 3;
STOP_SIGN = 4;

%% Initialize QLabs and QCar2
fprintf('Initializing QLabs/QUARC connection...\n');

% Add QVL MATLAB library path if QAL_DIR is set
qal_dir = getenv('QAL_DIR');
if ~isempty(qal_dir)
    qvl_path = fullfile(qal_dir, '0_libraries', 'matlab', 'qvl');
    if exist(qvl_path, 'dir')
        addpath(qvl_path);
        fprintf('Added QVL path: %s\n', qvl_path);
    end
end

try
    % Check if QUARC is available (needed for vehicle actuation)
    % The self-driving stack uses Simulink + QUARC for real control.
    % This script provides a standalone control loop for testing.
    quarc_available = exist('qc_stop_model', 'file') == 2;

    if ~quarc_available
        fprintf('WARNING: QUARC not found on MATLAB path.\n');
        fprintf('Running in simulation mode with mock vehicle.\n');
        fprintf('For real vehicle control, use VIRTUAL_self_driving_stack_v2.slx\n');
        use_mock_vehicle = true;
    else
        use_mock_vehicle = false;
        fprintf('QUARC detected. Vehicle control available.\n');
        % In production, vehicle is controlled via Simulink model.
        % For standalone mode, QUARC QCar2 API would be used here.
    end

catch ME
    fprintf('Error initializing QLabs: %s\n', ME.message);
    fprintf('Running in simulation mode.\n');
    use_mock_vehicle = true;
end

%% Initialize UDP Communication (Java, no toolboxes)
fprintf('Starting UDP communication (Java)...\n');

localPort = config.udp.local_port;   % 5005
[sock, sockCleanup] = udp_recv_java(localPort, 50);  % 50 ms timeout
fprintf('UDP receiver listening on port %d\n', localPort);

last_json = "";   % keep last valid message as fallback

%% Load or Define Waypoints
% Define a simple circular path for testing
% In production, this would be loaded from the competition map

fprintf('Loading waypoints...\n');

% Example waypoints (circular path)
num_points = 100;
theta = linspace(0, 2*pi, num_points);
radius = 2.0;  % meters

path = struct();
path.x = radius * cos(theta);
path.y = radius * sin(theta);

% Calculate path headings
path.heading = zeros(1, num_points);
for i = 1:num_points-1
    dx = path.x(i+1) - path.x(i);
    dy = path.y(i+1) - path.y(i);
    path.heading(i) = atan2(dy, dx);
end
path.heading(end) = path.heading(end-1);

fprintf('Loaded %d waypoints\n', num_points);

%% Initialize State
vehicle_state = struct();
vehicle_state.x = path.x(1);
vehicle_state.y = path.y(1);
vehicle_state.heading = path.heading(1);
vehicle_state.speed = 0;

control_state = struct();
control_state.current_state = CRUISE;
control_state.state_entry_time = 0;
control_state.target_speed = config.speed.cruise_speed;

% Data logging
log = struct();
log.time = [];
log.x = [];
log.y = [];
log.heading = [];
log.speed = [];
log.steering = [];
log.throttle = [];
log.state = [];
log.cross_track_error = [];

%% Main Control Loop
fprintf('\n=== Starting Main Control Loop ===\n');
fprintf('Press Ctrl+C to stop\n\n');

start_time = tic;
loop_count = 0;

try
    while toc(start_time) < config.sim.max_time

        loop_start = tic;
        current_time = toc(start_time);

        %% 1. Receive Vision Data from Python (Java UDP)
        [msg, ok] = udp_read_java(sock, 8192);

        if ok
            last_json = msg;
        end

        vision_msg = [];
        if strlength(last_json) > 0
            try
                vision_msg = jsondecode(last_json);
            catch
                vision_msg = [];  % ignore malformed packets
            end
        end

        %% 2. State Machine Update
        if ~isempty(vision_msg)
            [control_state, target_speed] = update_state_machine(...
                control_state, vision_msg, config, current_time, ...
                CRUISE, STOP_RED, STOP_PERSON, PICKUP_WAIT);
        else
            target_speed = 0;  % Stop if no vision data
        end

        %% 3. Get Vehicle State
        if use_mock_vehicle
            % Simulate vehicle motion
            % In production, this comes from QCar sensors/EKF
        else
            % Read from QCar
            % [vehicle_state.x, vehicle_state.y, vehicle_state.heading] = qcar.getPose();
            % vehicle_state.speed = qcar.getSpeed();
        end

        %% 4. Stanley Steering Controller
        [steering, debug_info] = stanley_controller(vehicle_state, path, config.stanley);

        % Optionally blend with vision-based steering suggestion
        if ~isempty(vision_msg) && vision_msg.lane_detected
            % Blend Stanley and vision steering (Stanley dominant)
            alpha = 0.8;  % Weight for Stanley controller
            steering = alpha * steering + (1 - alpha) * vision_msg.steering_suggestion;
        end

        %% 5. Speed Controller
        [throttle, brake] = speed_controller(vehicle_state.speed, target_speed, config.speed);

        % Override throttle in stop states
        if control_state.current_state == STOP_RED || ...
           control_state.current_state == STOP_PERSON || ...
           control_state.current_state == PICKUP_WAIT
            throttle = 0;
            brake = 1.0;  % Apply brakes
        end

        %% 6. Apply Control to Vehicle
        if use_mock_vehicle
            % Simple kinematic simulation
            dt = config.sim.dt;

            % Update speed
            if throttle > 0
                vehicle_state.speed = vehicle_state.speed + throttle * dt * 2;
            end
            if brake > 0
                vehicle_state.speed = max(0, vehicle_state.speed - brake * dt * 3);
            end
            vehicle_state.speed = min(vehicle_state.speed, 0.5);  % Max speed

            % Update position
            vehicle_state.x = vehicle_state.x + vehicle_state.speed * cos(vehicle_state.heading) * dt;
            vehicle_state.y = vehicle_state.y + vehicle_state.speed * sin(vehicle_state.heading) * dt;

            % Update heading
            L = 0.2;  % Wheelbase
            if vehicle_state.speed > 0.01
                vehicle_state.heading = vehicle_state.heading + ...
                    (vehicle_state.speed / L) * tan(steering) * dt;
            end
        else
            % Send to QCar
            % qcar.setThrottle(throttle);
            % qcar.setSteering(steering);
        end

        %% 7. Send Feedback to Python (via Java UDP - no toolboxes)
        if ~isempty(vision_msg)
            feedback = struct();
            feedback.timestamp = current_time;
            feedback.current_state = control_state.current_state;
            feedback.throttle = throttle;
            feedback.steering = steering;
            feedback.speed = vehicle_state.speed;
            feedback.x = vehicle_state.x;
            feedback.y = vehicle_state.y;
            feedback.heading = vehicle_state.heading;
            feedback.last_vision_frame_id = vision_msg.frame_id;

            json_str = jsonencode(feedback);
            udp_send_java(json_str, '127.0.0.1', config.udp.remote_port);
        end

        %% 8. Logging
        log.time(end+1) = current_time;
        log.x(end+1) = vehicle_state.x;
        log.y(end+1) = vehicle_state.y;
        log.heading(end+1) = vehicle_state.heading;
        log.speed(end+1) = vehicle_state.speed;
        log.steering(end+1) = steering;
        log.throttle(end+1) = throttle;
        log.state(end+1) = control_state.current_state;
        log.cross_track_error(end+1) = debug_info.cross_track_error;

        %% 9. Display Status (every 30 iterations)
        loop_count = loop_count + 1;
        if mod(loop_count, 30) == 0
            state_names = {'CRUISE', 'STOP_RED', 'STOP_PERSON', 'PICKUP_WAIT', 'STOP_SIGN'};
            fprintf('T=%.1fs | State: %s | Speed: %.2f | Steer: %.3f | CTE: %.3f\n', ...
                    current_time, state_names{control_state.current_state + 1}, ...
                    vehicle_state.speed, steering, debug_info.cross_track_error);
        end

        %% 10. Timing
        loop_time = toc(loop_start);
        pause_time = config.sim.dt - loop_time;
        if pause_time > 0
            pause(pause_time);
        end
    end

catch ME
    fprintf('\nControl loop interrupted: %s\n', ME.message);
end

%% Cleanup
fprintf('\n=== Stopping Control System ===\n');

% Close UDP socket (handled automatically by sockCleanup onCleanup object)
clear sockCleanup;

% Stop vehicle
if ~use_mock_vehicle
    % qcar.setThrottle(0);
    % qcar.setSteering(0);
end

%% Plot Results
fprintf('Plotting results...\n');

figure('Name', 'QCar Control Results', 'Position', [100 100 1200 800]);

% Path and trajectory
subplot(2, 3, 1);
plot(path.x, path.y, 'b--', 'LineWidth', 1);
hold on;
plot(log.x, log.y, 'r-', 'LineWidth', 1.5);
plot(log.x(1), log.y(1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(log.x(end), log.y(end), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('X (m)');
ylabel('Y (m)');
title('Path vs Trajectory');
legend('Path', 'Trajectory', 'Start', 'End');
axis equal;
grid on;

% Speed
subplot(2, 3, 2);
plot(log.time, log.speed, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Speed (m/s)');
title('Vehicle Speed');
grid on;

% Steering
subplot(2, 3, 3);
plot(log.time, rad2deg(log.steering), 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Steering (deg)');
title('Steering Angle');
grid on;

% Cross-track error
subplot(2, 3, 4);
plot(log.time, log.cross_track_error, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('CTE (m)');
title('Cross-Track Error');
grid on;

% State
subplot(2, 3, 5);
stairs(log.time, log.state, 'b-', 'LineWidth', 1.5);
yticks([0 1 2 3 4]);
yticklabels({'CRUISE', 'STOP\_RED', 'STOP\_PERSON', 'PICKUP\_WAIT', 'STOP\_SIGN'});
xlabel('Time (s)');
title('Vehicle State');
grid on;

% Throttle
subplot(2, 3, 6);
plot(log.time, log.throttle, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Throttle');
title('Throttle Command');
grid on;

fprintf('\nControl session complete.\n');
fprintf('Total runtime: %.1f seconds\n', log.time(end));
fprintf('Total loop iterations: %d\n', loop_count);

%% Helper Function: State Machine Update
function [control_state, target_speed] = update_state_machine(control_state, vision_msg, config, current_time, CRUISE, STOP_RED, STOP_PERSON, PICKUP_WAIT)
% Update state machine based on vision inputs

    prev_state = control_state.current_state;

    switch control_state.current_state
        case CRUISE
            % Normal driving
            target_speed = config.speed.cruise_speed;

            % Check for red light
            if vision_msg.red_light
                control_state.current_state = STOP_RED;
                control_state.state_entry_time = current_time;
                fprintf('State -> STOP_RED (red light detected)\n');
            % Check for person in pickup zone
            elseif vision_msg.person_in_pickup_zone
                control_state.current_state = STOP_PERSON;
                control_state.state_entry_time = current_time;
                fprintf('State -> STOP_PERSON (pedestrian detected)\n');
            end

        case STOP_RED
            % Stopped at red light
            target_speed = 0;

            % Check for green light
            if vision_msg.green_light
                control_state.current_state = CRUISE;
                control_state.state_entry_time = current_time;
                fprintf('State -> CRUISE (green light)\n');
            end

        case STOP_PERSON
            % Stopping for pedestrian
            target_speed = 0;

            % Transition to pickup wait after brief stop
            elapsed = current_time - control_state.state_entry_time;
            if elapsed > 0.5
                control_state.current_state = PICKUP_WAIT;
                control_state.state_entry_time = current_time;
                fprintf('State -> PICKUP_WAIT\n');
            end

        case PICKUP_WAIT
            % Waiting during pickup
            target_speed = 0;

            % Resume after wait time
            elapsed = current_time - control_state.state_entry_time;
            if elapsed >= config.state.pickup_wait_time
                control_state.current_state = CRUISE;
                control_state.state_entry_time = current_time;
                fprintf('State -> CRUISE (pickup complete)\n');
            end

        otherwise
            target_speed = 0;
    end
end
