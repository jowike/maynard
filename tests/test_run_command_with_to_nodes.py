"""
Test CLI command execution: Ensure the run command works as expected with valid inputs.
"""

import subprocess

def test_run_command_with_to_nodes():
    result = subprocess.run(
        ["maynard", "run", "--to-nodes", "estimate_ml_models,estimate_arima,estimate_var"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Expected success, but got {result.returncode}"
    assert result.stderr == ''
