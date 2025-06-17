"""
Test mutually exclusive parameters: Ensure the error is raised when multiple conflicting arguments are passed.
"""

import subprocess

def test_run_command_with_multiple_flags():
    result = subprocess.run(
        ["maynard", "run", "--from-nodes", "transform_time_series", "--to-nodes", "estimate_ml_models,estimate_arima,estimate_var"],
        capture_output=True,
        text=True
    )
    assert result.returncode != 0, "Expected failure when passing multiple conflicting options"