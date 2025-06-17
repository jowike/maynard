import subprocess
import time
import os
import signal
import socket

def is_port_open(host: str, port: int) -> bool:
    """Check if the given port is open on the specified host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # Timeout for the connection attempt
    try:
        sock.connect((host, port))
        sock.close()
        return True
    except (socket.timeout, socket.error):
        return False

def test_viz_command():

    def __find_free_port(port=5001, max_port=65535):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while port <= max_port:
            try:
                sock.bind(("", port))
                sock.close()
                return port
            except OSError:
                port += 1
        raise IOError("no free ports")
    

    viz_port = __find_free_port()

    kedro_viz_process = subprocess.Popen(
        f"maynard viz --port={viz_port}",  # Run the command in the shell
        shell=True,  # Use the shell to execute the command
        preexec_fn=os.setsid,  # Create a new process group
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr
        text=True
    )

    # Give the server some time to start (you can adjust this as necessary)
    time.sleep(30)  # Adjust this based on how long it typically takes for Kedro Viz to start

    # Check if Kedro Viz is running on the expected port (e.g., 5001)
    if is_port_open("127.0.0.1", viz_port):
        print(f"Kedro Viz is running on port {viz_port}")
    else:
        print(f"Kedro Viz did not start on port {viz_port}")
        # If the server is not running, terminate the process and fail the test
        os.killpg(kedro_viz_process.pid, signal.SIGTERM)
        kedro_viz_process.wait()
        assert False, "Kedro Viz did not start successfully"

    # Terminate the process group after testing
    os.killpg(kedro_viz_process.pid, signal.SIGTERM)  # Sends SIGTERM to the entire process group

    # Ensure the process has terminated
    kedro_viz_process.wait()

