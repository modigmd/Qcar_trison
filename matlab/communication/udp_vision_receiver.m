classdef udp_vision_receiver < handle
% UDP_VISION_RECEIVER - Receives vision messages from Python over UDP
%
% This class handles UDP communication with the Python vision system.
% It receives JSON-encoded messages containing:
%   - Traffic light detection (red, green, yellow)
%   - Person detection (presence, in pickup zone)
%   - Lane detection (steering suggestion)
%   - Suggested vehicle state
%
% Usage:
%   receiver = udp_vision_receiver();
%   receiver.start();
%   while running
%       msg = receiver.receive();
%       if ~isempty(msg)
%           % Process message
%       end
%   end
%   receiver.stop();
%
% Author: QCar AV System

    properties
        % UDP Configuration
        LocalPort = 5005;       % Port to listen on (matches Python matlab_port)
        RemoteHost = '127.0.0.1';
        RemotePort = 5006;      % Port to send feedback to Python
        BufferSize = 4096;
        Timeout = 0.01;         % Non-blocking timeout (seconds)

        % UDP Socket
        Socket = [];

        % State
        IsRunning = false;

        % Statistics
        MessagesReceived = 0;
        LastMessageTime = 0;

        % Last received message
        LastMessage = [];
    end

    methods
        function obj = udp_vision_receiver(local_port, remote_host, remote_port)
            % Constructor
            %
            % Args:
            %   local_port  - Port to listen on (default: 5005)
            %   remote_host - Python host IP (default: '127.0.0.1')
            %   remote_port - Python port for feedback (default: 5006)

            if nargin >= 1 && ~isempty(local_port)
                obj.LocalPort = local_port;
            end
            if nargin >= 2 && ~isempty(remote_host)
                obj.RemoteHost = remote_host;
            end
            if nargin >= 3 && ~isempty(remote_port)
                obj.RemotePort = remote_port;
            end
        end

        function success = start(obj)
            % Start the UDP receiver
            %
            % Returns:
            %   success - true if started successfully

            try
                % Create UDP object
                obj.Socket = udpport("datagram", "LocalPort", obj.LocalPort, ...
                    "Timeout", obj.Timeout, "EnablePortSharing", true);

                obj.IsRunning = true;
                obj.MessagesReceived = 0;

                fprintf('UDP Vision Receiver started on port %d\n', obj.LocalPort);
                success = true;

            catch ME
                fprintf('Error starting UDP receiver: %s\n', ME.message);
                success = false;
            end
        end

        function stop(obj)
            % Stop the UDP receiver

            obj.IsRunning = false;

            if ~isempty(obj.Socket)
                try
                    clear obj.Socket;
                catch
                    % Ignore errors during cleanup
                end
                obj.Socket = [];
            end

            fprintf('UDP Vision Receiver stopped. Total messages: %d\n', obj.MessagesReceived);
        end

        function msg = receive(obj)
            % Receive a vision message (non-blocking)
            %
            % Returns:
            %   msg - struct with vision data, or empty if no message

            msg = [];

            if ~obj.IsRunning || isempty(obj.Socket)
                return;
            end

            try
                % Check if data available
                if obj.Socket.NumDatagramsAvailable > 0
                    % Read datagram
                    data = read(obj.Socket, 1, "string");

                    if ~isempty(data)
                        % Parse JSON
                        msg = obj.parseMessage(data.Data);

                        if ~isempty(msg)
                            obj.MessagesReceived = obj.MessagesReceived + 1;
                            obj.LastMessageTime = posixtime(datetime('now'));
                            obj.LastMessage = msg;
                        end
                    end
                end

            catch ME
                % Ignore timeout errors, report others
                if ~contains(ME.message, 'timeout', 'IgnoreCase', true)
                    fprintf('Error receiving message: %s\n', ME.message);
                end
            end
        end

        function msg = receiveLatest(obj)
            % Receive the most recent message, discarding older ones
            %
            % Returns:
            %   msg - most recent message or empty

            msg = [];

            while true
                newMsg = obj.receive();
                if isempty(newMsg)
                    break;
                end
                msg = newMsg;
            end
        end

        function success = sendFeedback(obj, control_msg)
            % Send feedback message to Python
            %
            % Args:
            %   control_msg - struct with control state
            %
            % Returns:
            %   success - true if sent successfully

            success = false;

            if ~obj.IsRunning || isempty(obj.Socket)
                return;
            end

            try
                % Convert to JSON
                json_str = jsonencode(control_msg);

                % Send via UDP
                write(obj.Socket, json_str, "string", obj.RemoteHost, obj.RemotePort);

                success = true;

            catch ME
                fprintf('Error sending feedback: %s\n', ME.message);
            end
        end

        function msg = getLastMessage(obj)
            % Get the last received message
            msg = obj.LastMessage;
        end

        function delete(obj)
            % Destructor - ensure socket is closed
            obj.stop();
        end
    end

    methods (Access = private)
        function msg = parseMessage(obj, json_str)
            % Parse JSON message into struct
            %
            % Args:
            %   json_str - JSON string
            %
            % Returns:
            %   msg - parsed struct or empty

            msg = [];

            try
                msg = jsondecode(json_str);

                % Ensure all expected fields exist with defaults
                msg = ensureField(msg, 'timestamp', 0);
                msg = ensureField(msg, 'red_light', false);
                msg = ensureField(msg, 'green_light', false);
                msg = ensureField(msg, 'yellow_light', false);
                msg = ensureField(msg, 'traffic_light_confidence', 0);
                msg = ensureField(msg, 'person_detected', false);
                msg = ensureField(msg, 'person_in_pickup_zone', false);
                msg = ensureField(msg, 'person_confidence', 0);
                msg = ensureField(msg, 'lane_detected', false);
                msg = ensureField(msg, 'steering_suggestion', 0);
                msg = ensureField(msg, 'lane_center_offset', 0);
                msg = ensureField(msg, 'suggested_state', 0);
                msg = ensureField(msg, 'frame_id', 0);

            catch ME
                fprintf('Error parsing JSON: %s\n', ME.message);
            end
        end
    end
end

function s = ensureField(s, field, default)
% Ensure a field exists in struct with default value
    if ~isfield(s, field)
        s.(field) = default;
    end
end
