function ok = udp_send_java(jsonStr, remoteHost, remotePort)
% UDP_SEND_JAVA - Send a JSON string via Java DatagramSocket (no toolboxes)
%
% Inputs:
%   jsonStr    - string or char array to send
%   remoteHost - destination IP (e.g. '127.0.0.1')
%   remotePort - destination port (e.g. 5006)
%
% Outputs:
%   ok - true if sent successfully
%
% Author: QCar AV System

import java.net.DatagramSocket
import java.net.DatagramPacket
import java.net.InetAddress

ok = false;

try
    sendSock = DatagramSocket();
    addr = InetAddress.getByName(remoteHost);
    payload = uint8(char(jsonStr));
    pkt = DatagramPacket(payload, length(payload), addr, remotePort);
    sendSock.send(pkt);
    sendSock.close();
    ok = true;
catch ME
    fprintf('udp_send_java error: %s\n', ME.message);
end
end
