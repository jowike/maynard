import subprocess
import tempfile
from pathlib import Path
import time
import socket
import os
import signal


def is_port_open(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False


def find_free_port(start=5001, end=6000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        for port in range(start, end):
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise IOError("No free port available")


def test_viz_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "project"

        print("\n" + "=" * 30)
        print("🏗️  Step 1: Initializing project for Viz test")
        print("=" * 30)
        print(f"Project path: {project_path}")

        subprocess.run(["maynard", "init", str(project_path)], check=True)

        print("\n" + "=" * 30)
        print("📊 Step 2: Starting Kedro Viz")
        print("=" * 30)

        viz_port = find_free_port()
        viz_command = f"maynard viz --port={viz_port}"
        print(f"Running command: {viz_command}")

        proc = subprocess.Popen(
            viz_command,
            shell=True,
            cwd=project_path,  # Must run inside valid project
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # Avoid PermissionError on some OS
        )

        time.sleep(15)  # Give the server time to start

        print("\n🧪 Verifying if Kedro Viz is running...")

        if not is_port_open("127.0.0.1", viz_port):
            stdout, stderr = proc.communicate(timeout=10)
            print("\n[STDOUT] Kedro Viz:\n", stdout.strip())
            print("\n[STDERR] Kedro Viz:\n", stderr.strip())
            proc.terminate()
            assert False, f"Kedro Viz did not start on port {viz_port}"

        print(f"\n✅ Kedro Viz successfully started on port {viz_port}")

        proc.terminate()
        proc.wait(timeout=10)
