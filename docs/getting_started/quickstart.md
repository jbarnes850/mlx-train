# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

```bash
# Clone and setup
git clone https://github.com/jbarnes850/mlx-train
cd mlx-train
./scripts/mlx-train.sh
```

## 🎯 What to Expect

The interactive script will guide you through 8 simple steps:

1. **Environment Setup** 

```bash
📦 Setting up environment...
✓ Python dependencies installed
✓ MLX configured for your device
```

2. **Hardware Detection**

```bash
🔍 Detected Hardware:
• Device: M2 Max
• Memory: 32GB
• Performance: 15.8 TFLOPS
```

3. **Model Selection**

```bash
🧠 Choose your model:
1. Transformer (Recommended)
2. MLP (Simple & Fast)
3. Custom Architecture
```

4. **Dataset Configuration**

```bash
📚 Select dataset:
1. Synthetic (for testing)
2. HuggingFace datasets
3. Custom dataset
```

5. **Training Setup**

```bash
⚙️ Optimal Configuration:
• Batch Size: 32
• Learning Rate: 3e-4
• Mixed Precision: Enabled
```

## 💫 Features & Capabilities

### Hardware Optimization

| Device      | Memory | Optimal Batch Size | Training Speed |
|-------------|--------|-------------------|----------------|
| M1 Pro/Max  | 16GB   | 32               | 1,000 tok/s    |
| M2 Pro/Max  | 32GB   | 64               | 1,500 tok/s    |
| M3 Pro/Max  | 48GB   | 96               | 2,000 tok/s    |

### Distributed Training

```bash
# Example: 2-device setup
Device 1 (M2 Max):    [==================] 80%
Device 2 (M2 Pro):    [==================] 75%
Combined TFLOPS: 30.8
```

### Real-time Monitoring

```bash
Training Progress: [==================] 50%
• Loss: 0.0234
• Accuracy: 94.5%
• Speed: 1,250 tokens/sec
• Memory: 25.6GB
```

## 🔧 Common Configurations

### Simple Testing

```bash
# Quick test setup
./scripts/mlx-train.sh --preset test
```

### Production Training

```bash
# Full training setup
./scripts/mlx-train.sh --preset production
```

### Distributed Setup

```bash
# Multi-device training
./scripts/mlx-train.sh --distributed
```

## 📊 Performance Benchmarks

| Model Size | Devices | Throughput     | Memory Usage |
|------------|---------|----------------|--------------|
| 768M       | 1       | 1,000 tok/s    | 16 GB       |
| 768M       | 2       | 1,800 tok/s    | 8 GB/device |
| 1.5B       | 1       | 500 tok/s      | 32 GB       |
| 1.5B       | 2       | 900 tok/s      | 16 GB/device|

## 🎯 Best Practices

1. **Memory Management**
   - Start with smaller batch sizes
   - Enable mixed precision
   - Monitor memory usage

2. **Distributed Training**
   - Use similar devices when possible
   - Ensure stable network connection
   - Start with data parallel approach

3. **Model Development**
   - Begin with provided templates
   - Use incremental testing
   - Enable checkpointing

## 🔍 Troubleshooting

Common issues and solutions:

1. **Memory Errors**

```bash
# Reduce batch size
./scripts/mlx-train.sh --batch-size 16
```

2. **Network Issues**

```bash
# Check connectivity
./scripts/mlx-train.sh --check-network
```

3. **Performance Issues**

```bash
# Enable optimizations
./scripts/mlx-train.sh --optimize
```

## 📚 Next Steps

- [Model Development](model_development.md)
- [Distributed Setup](distributed_setup.md)
