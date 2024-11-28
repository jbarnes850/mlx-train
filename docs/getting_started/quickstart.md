# Quick Start Guide

This guide will help you get started with MLX Distributed Training quickly.

## Initialize Your Project

Create a new MLX training project:

```bash
mlx-train init my-project
cd my-project
```

Your project will be initialized with the following structure:

```bash
my-project/
├── config/
│   └── default.json     # Training configuration
├── data/
│   └── README.md       # Data management instructions
├── models/
│   └── custom.py      # Custom model definitions
├── experiments/
│   └── README.md      # Experiment tracking
└── README.md          # Project documentation
```

## Configure Training

The framework provides an interactive configuration process:

```bash
mlx-train configure
```

You'll be guided through:

1. **Model Selection**
   - Small (768M parameters, good for testing)
   - Medium (1B parameters, balanced)
   - Large (2B+ parameters, distributed training)
   - Custom configuration

2. **Hardware Configuration**
   - Automatic device detection
   - Memory optimization
   - Distributed setup (if multiple devices)

3. **Training Parameters**

   ```bash
   # Example configuration
   mlx-train configure \
     --model-size medium \
     --batch-size 32 \
     --learning-rate 1e-4 \
     --distributed  # if multiple devices
   ```

## Prepare Dataset

Choose your training data:

```bash
# Use Hugging Face dataset
mlx-train data fetch --dataset huggingface/dataset-name

# Use local data
mlx-train data prepare --source path/to/data
```

Supported formats:

- Text files (.txt)
- CSV files (.csv)
- JSON/JSONL files (.json, .jsonl)
- Parquet files (.parquet)

## Start Training

Launch training with real-time monitoring:

```bash
mlx-train train
```

You'll see:

```bash
🚀 Training Progress
├── Loss: 2.345 → 1.234
├── Speed: 256 samples/sec
├── Memory: 75% utilized
└── Devices: 2 active

Press Ctrl+C to stop training
```

## Monitor Progress

View training metrics:

```bash
# View live metrics
mlx-train monitor

# Check device status
mlx-train status
```

## Export Model

Once training is complete, export your model:

```bash
# Export to MLX format
mlx-train export --format mlx

# Export for Ollama
mlx-train export --format gguf
```

## Serve Model

Start the model server:

```bash
mlx-train serve --port 8000
```

Test inference:

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'
```

## Next Steps

- [Distributed Training Guide](../guides/distributed_training.md)
- [Model Development Guide](../guides/model_development.md)
- [Memory Optimization Guide](../guides/memory_optimization.md)

## Troubleshooting

Common issues:

1. **Memory Errors**

   ```bash
   # Reduce batch size
   mlx-train train --batch-size 16
   ```

2. **Device Issues**

   ```bash
   # Check device status
   mlx-train check-devices
   ```

3. **Training Issues**

   ```bash
   # Enable debug mode
   MLX_DEBUG=1 mlx-train train
   ```

For more help, see our [Troubleshooting Guide](../guides/troubleshooting.md).
