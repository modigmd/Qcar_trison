function [sock, cleanupObj] = udp_recv_java(localPort, timeoutMs)
% UDP receiver using Java DatagramSocket (no toolboxes required)

import java.net.DatagramSocket
import java.net.DatagramPacket
import java.net.SocketTimeoutException

sock = DatagramSocket(localPort);
sock.setSoTimeout(timeoutMs);  % receive timeout (ms)

cleanupObj = onCleanup(@() safeClose(sock));

end

function safeClose(sock)
try
    sock.close();
catch
end
end
