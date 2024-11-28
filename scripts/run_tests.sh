#!/bin/bash

# Set environment variables
export PYTHONPATH="."
export MLX_DISTRIBUTED="1"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up MLX Training Framework Test Environment...${NC}"

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${BLUE}📥 Installing dependencies...${NC}"
pip install --upgrade pip
pip install -e ".[test]"

# Define test modes
run_basic_tests() {
    echo -e "${BLUE}🧪 Running basic functionality tests...${NC}"
    pytest \
        tests/test_core.py \
        tests/test_training.py \
        tests/test_data.py \
        -v \
        --capture=no \
        --log-cli-level=INFO
}

run_distributed_tests() {
    echo -e "${BLUE}🌐 Running distributed tests...${NC}"
    pytest \
        tests/test_distributed.py \
        tests/test_benchmarks.py \
        tests/test_export.py \
        -v \
        -m "distributed or export" \
        --capture=no \
        --log-cli-level=INFO
}

run_critical_tests() {
    echo -e "${BLUE}🔍 Running critical path tests...${NC}"
    pytest \
        tests/test_distributed.py::test_device_discovery \
        tests/test_distributed.py::test_training_resumption \
        tests/test_distributed.py::test_memory_optimization \
        tests/test_export.py::test_model_export \
        -v \
        --capture=no \
        --log-cli-level=INFO
}

run_full_tests() {
    echo -e "${BLUE}🔍 Running full test suite with coverage...${NC}"
    pytest \
        --verbose \
        --capture=no \
        --log-cli-level=INFO \
        --cov=mlx_train \
        --cov-report=term-missing \
        tests/
}

# Clean up function
cleanup() {
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    deactivate
}

# Parse command line arguments
case "${1:-full}" in
    "basic")
        run_basic_tests
        ;;
    "distributed")
        run_distributed_tests
        ;;
    "critical")
        run_critical_tests
        ;;
    "full")
        run_full_tests
        ;;
    *)
        echo "Usage: $0 [full|basic|distributed|critical]"
        echo "  full:        Run full test suite with coverage (default)"
        echo "  basic:       Run only basic functionality tests"
        echo "  distributed: Run distributed and benchmark tests"
        echo "  critical:    Run critical path tests only"
        exit 1
        ;;
esac

# Set up trap to clean up on exit
trap cleanup EXIT

echo -e "${GREEN}✨ Tests completed!${NC}" 