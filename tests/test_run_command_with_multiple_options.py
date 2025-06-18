"""
Test mutually exclusive parameters: Ensure the error is raised when multiple conflicting arguments are passed.
"""

import subprocess
import tempfile
from pathlib import Path


def test_run_command_with_multiple_flags():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        project_path = tmp_path / "project"

        print("\n" + "=" * 30)
        print("🏗️  Step 1: Initializing project")
        print("=" * 30)
        print(f"Project path: {project_path}")

        result_init = subprocess.run(
            ["maynard", "init", str(project_path)], capture_output=True, text=True
        )
        print("\n[STDOUT] Project Initialization:\n", result_init.stdout.strip())
        print("\n[STDERR] Project Initialization:\n", result_init.stderr.strip())
        assert result_init.returncode == 0, f"Init failed: {result_init.stderr}"

        print("\n" + "=" * 30)
        print("⚙️  Step 2: Running pipeline with multiple flags")
        print("=" * 30)
        print("From node: transform_time_series")
        print("To nodes: estimate_ml_models, estimate_arima, estimate_var")

        result_run = subprocess.run(
            [
                "maynard",
                "run",
                "--from-nodes",
                "transform_time_series",
                "--to-nodes",
                "estimate_ml_models,estimate_arima,estimate_var",
            ],
            cwd=project_path,
            capture_output=True,
            text=True,
        )
        print("\n[STDOUT] Pipeline Run:\n", result_run.stdout.strip())
        print("\n[STDERR] Pipeline Run:\n", result_run.stderr.strip())
        assert (
            result_run.returncode != 0
        ), f"Run should fail but succeeded: {result_run.stderr}"
