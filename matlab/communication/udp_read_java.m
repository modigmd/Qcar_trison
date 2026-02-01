function [msg, ok] = udp_read_java(sock, maxBytes)
% Read one UDP datagram (string) from Java DatagramSocket

import java.net.DatagramPacket
import java.net.SocketTimeoutException

ok = false;
msg = "";

buf = int8(zeros(1, maxBytes));
pkt = DatagramPacket(buf, maxBytes);

try
    sock.receive(pkt);
    n = pkt.getLength();
    data = typecast(pkt.getData(), 'uint8');
    msg = string(char(data(1:n)));
    ok = true;
catch ME
    % timeout is normal; return ok=false
    if ~contains(ME.message, 'Receive timed out') && ~contains(ME.message, 'SocketTimeoutException')
        rethrow(ME);
    end
end
end
