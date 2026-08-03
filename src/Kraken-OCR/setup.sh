#!/usr/bin/env bash
set -euo pipefail

# Script to setup Kraken-OCR environment with conda
# Automatically downloads and installs Miniconda if conda is not available
# Usage: bash setup.sh

PYTHON_VERSION="3.11"
ENV_NAME="kraken_ocr_env"
MINICONDA_VERSION="latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
  echo "❌ Requirements file not found: $REQ_FILE"
  exit 1
fi

echo "======================================"
echo "  Kraken-OCR Environment Setup"
echo "======================================"

# Function to install Miniconda
install_miniconda() {
  echo "📦 Conda not found. Installing Miniconda..."
  
  MINICONDA_DIR="$HOME/miniconda3"
  
  if [ -d "$MINICONDA_DIR" ]; then
    echo "⚠️  Miniconda directory already exists at $MINICONDA_DIR"
    echo "   Attempting to use existing installation..."
  else
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
      # Check if ARM or Intel Mac
      if [[ $(uname -m) == "arm64" ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-MacOSX-arm64.sh"
      else
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-MacOSX-x86_64.sh"
      fi
    else
      echo "❌ Unsupported operating system: $OSTYPE"
      echo "   Please install Miniconda manually from https://docs.conda.io/en/latest/miniconda.html"
      exit 1
    fi
    
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    
    echo "📥 Downloading Miniconda from $MINICONDA_URL..."
    curl -fsSL "$MINICONDA_URL" -o "$MINICONDA_INSTALLER"
    
    echo "🔧 Installing Miniconda to $MINICONDA_DIR..."
    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR"
    
    rm "$MINICONDA_INSTALLER"
    echo "✅ Miniconda installed successfully!"
  fi
  
  # Initialize conda for bash
  if [ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
  else
    echo "❌ Conda initialization script not found!"
    exit 1
  fi
  
  conda init bash
  echo "✅ Conda initialized. You may need to restart your shell or run 'source ~/.bashrc'"
}

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
  install_miniconda
fi

# Initialize conda in current shell
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  # Try to source from common locations
  for CONDA_PATH in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
      source "$CONDA_PATH/etc/profile.d/conda.sh"
      eval "$(conda shell.bash hook)"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ Conda still not available. Please restart your shell and run this script again."
  exit 1
fi

echo "✅ Conda is available"
echo "📋 Using Python version: $PYTHON_VERSION"

# Create environment if it doesn't exist
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "⏭️  Conda environment '$ENV_NAME' already exists. Skipping creation."
else
  echo "🔨 Creating conda environment '$ENV_NAME' with python=$PYTHON_VERSION..."
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION" pip
  echo "✅ Environment created successfully!"
fi

# Activate the environment
echo "🔄 Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Upgrade pip and install requirements
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

echo "📦 Installing requirements from $REQ_FILE..."
python -m pip install -r "$REQ_FILE"

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo "======================================"
echo "To activate the environment in a new shell, run:"
echo "  conda activate $ENV_NAME"
echo ""
