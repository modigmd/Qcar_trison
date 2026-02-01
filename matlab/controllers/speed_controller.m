function [throttle, brake] = speed_controller(current_speed, target_speed, params)
% SPEED_CONTROLLER - Simple speed controller with throttle/brake outputs
%
% Implements a P controller for speed regulation with separate
% throttle and brake commands.
%
% Inputs:
%   current_speed - current vehicle speed (m/s)
%   target_speed  - desired speed (m/s)
%   params        - struct with fields [optional]:
%       Kp           - proportional gain (default: 2.0)
%       max_throttle - maximum throttle (0-1) (default: 0.5)
%       max_brake    - maximum brake (0-1) (default: 1.0)
%       deadband     - speed error deadband (m/s) (default: 0.02)
%
% Outputs:
%   throttle - throttle command [0, 1]
%   brake    - brake command [0, 1]
%
% Author: QCar AV System

    %% Default parameters
    if nargin < 3 || isempty(params)
        params = struct();
    end

    Kp = get_param(params, 'Kp', 2.0);
    max_throttle = get_param(params, 'max_throttle', 0.5);
    max_brake = get_param(params, 'max_brake', 1.0);
    deadband = get_param(params, 'deadband', 0.02);

    %% Calculate speed error
    speed_error = target_speed - current_speed;

    %% Apply deadband
    if abs(speed_error) < deadband
        throttle = 0;
        brake = 0;
        return;
    end

    %% Calculate control output
    control = Kp * speed_error;

    %% Split into throttle and brake
    if control > 0
        % Need to accelerate
        throttle = min(control, max_throttle);
        brake = 0;
    else
        % Need to decelerate
        throttle = 0;
        brake = min(-control, max_brake);
    end
end

function val = get_param(params, name, default)
% Get parameter value with default
    if isfield(params, name)
        val = params.(name);
    else
        val = default;
    end
end
