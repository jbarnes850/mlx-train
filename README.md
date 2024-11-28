# 🚀 MLX Distributed Training Framework

Train and deploy AI models across multiple Apple Silicon devices with automatic hardware optimization and a seamless developer experience. A user-friendly CLI framework for distributed training, powered by MLX.

```bash
╔══════════════════════════════════════════════════════════╗
║  __  __ _     __  __  _____           _                  ║
║ |  \/  | |    \ \/ / |_   _| __ __ _ (_) _ __            ║
║ | |\/| | |     \  /    | | | '__/ _` || || '_ \          ║
║ | |  | | |___  /  \    | | | | | (_| || || | | |         ║
║ |_|  |_|_____|/_/\_\   |_| |_|  \__,_||_||_| |_|         ║
║                                                          ║
║          Distributed Training on Apple Silicon           ║
╚══════════════════════════════════════════════════════════╝

┌─────────────────┐      ┌─────────────────┐
│   MacBook Pro   │  ←→  │   MacBook Air   │  Training
│    Node 1       │      │    Node 2       │  Cluster
└────────┬────────┘      └────────┬────────┘
         │                        │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │    MLX Training      │  Distributed
         │   ═══════════ 100%   │  Progress
         └──────────────────────┘
                    ▼
         ┌──────────────────────┐
         │  🚀 Trained Model    │  Local
         │  localhost:8000      │  Deployment
         └──────────────────────┘
```

[![PyPI version](https://badge.fury.io/py/mlx-train.svg)](https://badge.fury.io/py/mlx-train)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚡️ Quick Start & Interactive Script

```bash
# Clone and start
git clone https://github.com/jbarnes850/mlx-train
cd mlx-train

# Run the interactive training script
./scripts/mlx-train.sh
```

That's it! The script will:

- 🛠️ Set up your environment automatically
- 🔍 Detect your Apple Silicon devices
- 🤝 Help connect multiple devices if available
- 🚀 Guide you through model training
- 🌐 Deploy your model locally

Watch your model train with real-time visualizations:

```bash
Device Utilization:
Device 1: [====================] 80.5% (12.8GB)
Device 2: [===================] 75.2% (12.1GB)

Training Progress:
[====================] 50%

Metrics:
• Throughput: 1250.32 tokens/sec
• Loss: 0.0234
• Memory Usage: 25.6GB
```

## ✨ Features

- 🚄 **Distributed Training**: Seamlessly scale across multiple Apple Silicon devices
- 🔧 **Hardware Optimization**: Automatic detection and configuration for optimal performance
- 🎯 **Zero-Config Setup**: Automatic environment setup and dependency management
- 📊 **Training Visualization**: Real-time metrics and progress tracking
- 🧠 **Model Development**: Build custom architectures or use pre-built components
- 🔄 **Export & Serve**: Deploy models locally or export for other platforms

**Scale When Ready**:

- 🔄 Start with any Apple Silicon Device (M1/M2/M3/M4)
- 🔄 Add more devices anytime
- 🔄 Automatic distributed training
- 🔄 No code changes needed

## 🛠 Installation & Requirements

### Requirements

- Python 3.10 or higher
- Apple Silicon Mac (M1/M2/M3)
- 8GB RAM minimum (16GB+ recommended)
- macOS 12 or higher

### Quick Install

```bash

pip install mlx-train
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/jbarnes850/mlx-train
cd mlx-train

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

The `mlx-train.sh` script will automatically check and install all requirements.

## 🔧 Advanced Configuration

### Hardware Optimization

- Automatic device discovery
- Memory-aware batch size selection
- Gradient accumulation optimization

### Training Options

- Mixed precision training
- Gradient checkpointing
- Custom learning rate schedules

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

[Documentation](docs/README.md)
[Installation](docs/installation.md)
[Examples](docs/examples.md)
[Core Concepts](docs/core.md)

## 🔬 Examples

Check out our [examples directory](examples/) for:

- Custom model training
- Multi-device distributed training
- Dataset preprocessing
- Model export and serving

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📫 Support

- 🐛 [Issue Tracker](https://github.com/jbarnes850/mlx-train/issues)

---
Made with ❤️ for the MLX community
