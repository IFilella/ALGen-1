import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generator import main
import subprocess

def test_main_generator():
    result = subprocess.run(
        [
            "python", "generator.py",
            "-g", "data/general_set.smi",
            "-e", "80",
            "-v", "0.1",
            "-t", "0.1",
            "-b", "16",
            "-s", "1.2",
            "-q", "100",
            "-n", "algen_run",
            "-o", "./results",
            "-pa", "data/specific_set.smi",
            "-ial", "5",
            "-qed", "0.6",
            "-sa", "5",
            "-ta", "0.6",
        ],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    print(result.stderr)
    
    # Check if it run successfully
    assert result.returncode == 0, "ALGen did not run successfully."

if __name__ == '__main__':
    test_main_generator()

