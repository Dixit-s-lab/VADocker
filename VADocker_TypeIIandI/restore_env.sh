#!/bin/bash
# =====================================================================
# Automated installation script for Conda environment: 'vadocker_amber'
# =====================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "-> Checking if Conda is installed..."
if ! command -v conda &> /dev/null; then
    echo "Error: Conda could not be found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "-> Creating Conda environment from 'environment.yml'..."
conda env create -f environment.yml

echo "====================================================================="
echo "Installation complete!"
echo "To activate your new environment, run:"
echo "    conda activate vadocker_amber"
echo "====================================================================="
