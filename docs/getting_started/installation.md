# Installation Guide

## Prerequisites

- macOS 13.3 or later (macOS 14 Sonoma recommended)
- Apple Silicon Mac (M1/M2/M3)
- Python 3.9+
- pip package manager

## Quick Install

Install via pip:

```bash
pip install mlx-train
```

## Development Install

For contributing or local development:

```bash
# Clone repository
git clone https://github.com/your-org/mlx-distributed-training
cd mlx-distributed-training

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -e ".[dev]"
```

## Verify Installation

Verify your installation:

```bash
mlx-train --version
```

## Dependencies

The framework automatically manages these dependencies:

### Core Dependencies

- MLX (>=0.21.0)
- MPI4Py (>=3.1.4)
- PyTorch (>=2.1.0)

### Optional Dependencies

- transformers: For using Hugging Face models
- datasets: For dataset management
- langchain: For text processing

## Troubleshooting

Common installation issues:

1. **MLX Installation Fails**

   ```bash
   # Try updating pip first
   python -m pip install --upgrade pip
   
   # Then reinstall MLX
   pip install --upgrade mlx
   ```

2. **MPI4Py Issues**

   ```bash
   # Install via Homebrew first
   brew install open-mpi
   pip install mpi4py
   ```

3. **Environment Issues**
   - Ensure you're using Python 3.9+
   - Verify you're on Apple Silicon
   - Check macOS version compatibility

## Next Steps

Once installed, see our [Quick Start Guide](quickstart.md) to begin training your first model.
