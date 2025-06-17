"""
Test CLI command execution: Ensure the run command works as expected with valid inputs.
"""
import subprocess

def test_run_command():
    import maynard
    
    result = subprocess.run(
        ["maynard", "run"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Expected success, but got {result.returncode}"
    assert result.stderr == ''
