import os
import subprocess
import sys

def main():
    # 1. Identify the current active Conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if not conda_env:
        print("Error: No active Conda environment detected. Please activate a Conda environment first.")
        sys.exit(1)
    
    print(f"[INFO] Detected active Conda environment: '{conda_env}'")

    env_file = "environment.yml"
    install_script_file = "restore_env.sh"

    # 2. Export the environment dependencies
    print(f"[INFO] Exporting dependencies to '{env_file}'...")
    try:
        # Using --no-builds makes the environment.yml more cross-platform compatible
        result = subprocess.run(
            ["conda", "env", "export", "--no-builds"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Clean up the output by removing the local machine's 'prefix:' line
        env_lines = result.stdout.splitlines()
        cleaned_lines = [line for line in env_lines if not line.startswith("prefix:")]
        
        with open(env_file, "w") as f:
            f.write("\n".join(cleaned_lines) + "\n")
            
        print(f"[SUCCESS] Created '{env_file}'")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to export conda environment: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 'conda' command not found. Make sure Conda is installed and accessible in your PATH.")
        sys.exit(1)

    # 3. Generate a convenient installation script for the target system
    print(f"[INFO] Generating installation script '{install_script_file}'...")
    install_script_content = f"""#!/bin/bash
# =====================================================================
# Automated installation script for Conda environment: '{conda_env}'
# =====================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "-> Checking if Conda is installed..."
if ! command -v conda &> /dev/null; then
    echo "Error: Conda could not be found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "-> Creating Conda environment from '{env_file}'..."
conda env create -f {env_file}

echo "====================================================================="
echo "Installation complete!"
echo "To activate your new environment, run:"
echo "    conda activate {conda_env}"
echo "====================================================================="
"""

    with open(install_script_file, "w") as f:
        f.write(install_script_content)
    
    # Make the generated shell script executable
    os.chmod(install_script_file, 0o755)
    print(f"[SUCCESS] Created '{install_script_file}'")

    # 4. Final instructions
    print("\n" + "="*50)
    print("READY FOR TRANSFER!")
    print("Move the following two files to your target system:")
    print(f"  1. {env_file}")
    print(f"  2. {install_script_file}")
    print("\nOn the target system, simply run:")
    print(f"  ./{install_script_file}")
    print("="*50)

if __name__ == "__main__":
    main()