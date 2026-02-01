function [steering, debug_info] = stanley_controller(vehicle_state, path, params)
% STANLEY_CONTROLLER - Stanley steering controller for lane following
%
% Implements the Stanley steering control law:
%   delta = theta_e + arctan(k * e_ct / (v + epsilon))
%
% Where:
%   delta   = steering angle
%   theta_e = heading error (angle between vehicle heading and path tangent)
%   k       = cross-track error gain
%   e_ct    = cross-track error (lateral distance to path)
%   v       = vehicle speed
%   epsilon = softening constant (prevents division by zero)
%
% Based on "Stanley: The Robot that Won the DARPA Grand Challenge"
%
% Inputs:
%   vehicle_state - struct with fields:
%       x       - x position (m)
%       y       - y position (m)
%       heading - heading angle (rad)
%       speed   - forward speed (m/s)
%
%   path - struct with fields:
%       x       - array of x waypoints (m)
%       y       - array of y waypoints (m)
%       heading - array of path headings (rad) [optional]
%
%   params - struct with fields [optional]:
%       k                   - cross-track error gain (default: 2.5)
%       epsilon             - softening constant (default: 0.1)
%       max_steering        - maximum steering angle (rad) (default: 0.5)
%       steering_rate_limit - max steering rate (rad/s) (default: 0.8)
%       lookahead_distance  - distance to lookahead point (m) (default: 0.1)
%
% Outputs:
%   steering   - steering angle command (rad)
%   debug_info - struct with debugging information
%
% Author: QCar AV System

    %% Default parameters
    if nargin < 3 || isempty(params)
        params = struct();
    end

    % Get parameters with defaults
    k = get_param(params, 'k', 2.5);
    epsilon = get_param(params, 'epsilon', 0.1);
    max_steering = get_param(params, 'max_steering', 0.5);
    lookahead_dist = get_param(params, 'lookahead_distance', 0.1);

    %% Extract vehicle state
    x = vehicle_state.x;
    y = vehicle_state.y;
    heading = vehicle_state.heading;
    speed = vehicle_state.speed;

    %% Find closest point on path
    [closest_idx, closest_dist, cross_track_error] = find_closest_point(x, y, path);

    %% Calculate path heading at closest point
    path_heading = calculate_path_heading(path, closest_idx);

    %% Calculate heading error
    heading_error = normalize_angle(path_heading - heading);

    %% Apply Stanley control law
    % Cross-track correction term
    cross_track_term = atan2(k * cross_track_error, speed + epsilon);

    % Total steering command
    steering = heading_error + cross_track_term;

    %% Apply limits
    steering = max(min(steering, max_steering), -max_steering);

    %% Debug information
    debug_info = struct();
    debug_info.closest_idx = closest_idx;
    debug_info.closest_dist = closest_dist;
    debug_info.cross_track_error = cross_track_error;
    debug_info.heading_error = heading_error;
    debug_info.path_heading = path_heading;
    debug_info.cross_track_term = cross_track_term;
    debug_info.raw_steering = heading_error + cross_track_term;
    debug_info.limited_steering = steering;
end

function val = get_param(params, name, default)
% Get parameter value with default
    if isfield(params, name)
        val = params.(name);
    else
        val = default;
    end
end

function [closest_idx, closest_dist, cross_track_error] = find_closest_point(x, y, path)
% Find the closest point on the path to the vehicle position
%
% Also calculates the signed cross-track error (positive = vehicle is to the right of path)

    % Calculate distances to all waypoints
    dx = path.x - x;
    dy = path.y - y;
    distances = sqrt(dx.^2 + dy.^2);

    % Find minimum
    [closest_dist, closest_idx] = min(distances);

    % Calculate signed cross-track error
    % Use cross product to determine sign
    if closest_idx < length(path.x)
        % Vector from closest point to next point (path direction)
        path_dx = path.x(closest_idx + 1) - path.x(closest_idx);
        path_dy = path.y(closest_idx + 1) - path.y(closest_idx);
    else
        % Use previous segment at end of path
        path_dx = path.x(closest_idx) - path.x(closest_idx - 1);
        path_dy = path.y(closest_idx) - path.y(closest_idx - 1);
    end

    % Vector from closest point to vehicle
    veh_dx = x - path.x(closest_idx);
    veh_dy = y - path.y(closest_idx);

    % Cross product z-component determines sign
    cross_z = path_dx * veh_dy - path_dy * veh_dx;

    % Signed error (positive = vehicle to right of path)
    cross_track_error = sign(cross_z) * closest_dist;
end

function path_heading = calculate_path_heading(path, idx)
% Calculate the path heading at a given index

    n = length(path.x);

    % Handle edge cases
    if n < 2
        path_heading = 0;
        return;
    end

    if idx >= n
        idx = n - 1;
    end
    if idx < 1
        idx = 1;
    end

    % Calculate heading from idx to idx+1
    dx = path.x(min(idx + 1, n)) - path.x(idx);
    dy = path.y(min(idx + 1, n)) - path.y(idx);

    path_heading = atan2(dy, dx);
end

function angle = normalize_angle(angle)
% Normalize angle to [-pi, pi]
    while angle > pi
        angle = angle - 2 * pi;
    end
    while angle < -pi
        angle = angle + 2 * pi;
    end
end
