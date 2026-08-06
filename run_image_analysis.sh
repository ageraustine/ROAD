#!/bin/bash
# Image Dataset Analysis Runner
# Uses .venv virtual environment to analyze images in dataset/images

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}  Image Dataset Analyzer for R.O.A.D. Competition${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: .venv directory not found${NC}"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating .venv...${NC}"
source .venv/bin/activate

# Check if required packages are installed
echo -e "${GREEN}Checking dependencies...${NC}"
if ! python -c "import PIL, numpy, tqdm" 2>/dev/null; then
    echo -e "${RED}Missing dependencies. Installing...${NC}"
    pip install pillow numpy tqdm
fi

# Run the analysis
echo -e "${GREEN}Running image analysis...${NC}"
echo
python analyze_images.py "$@"

echo
echo -e "${GREEN}Analysis complete!${NC}"
echo -e "${BLUE}Use these insights to update max_pixels in:${NC}"
echo "  - src/qwen2vl/config.yaml"
echo "  - src/qwen2vl/config_qwen3_2b.yaml"
