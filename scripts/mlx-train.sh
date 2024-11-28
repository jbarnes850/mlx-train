#!/bin/bash

# Set environment variables
export PYTHONPATH="."
export MLX_DISTRIBUTED="1"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

progress_bar() {
    local duration=$1
    local prefix=$2
    local size=40
    local progress=0
    
    while [ $progress -le $size ]; do
        echo -ne "\r${prefix} ["
        for ((i=0; i<$size; i++)); do
            if [ $i -lt $progress ]; then
                echo -ne "="
            else
                echo -ne " "
            fi
        done
        echo -ne "] $((progress*100/size))%"
        progress=$((progress+1))
        sleep $(echo "scale=3; $duration/$size" | bc)
    done
    echo
}

# Welcome
show_welcome() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║        🚀 MLX Training Framework               ║${NC}"
    echo -e "${BLUE}║     Train AI Models on Apple Silicon           ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
    echo -e "\n${GREEN}Welcome! Train your AI model from scratch.${NC}"
    echo -e "${YELLOW}This experience will help you:${NC}"
    echo -e "• 🛠️  Set up your distributed training environment"
    echo -e "• 🧠  Create your own AI model architecture"
    echo -e "• 📚  Choose and prepare your training data"
    echo -e "• 🚀  Train your model efficiently"
    echo -e "• 🌐  Deploy, share, and interact with your model"
    echo
    read -p "Press Enter to begin..."
}

show_tooltips() {
    echo -e "\n${BLUE}📚 Quick Reference:${NC}"
    echo -e "┌────────────────────────────────────────────┐"
    echo -e "│ ${YELLOW}Key Terms:${NC}                   │"
    echo -e "│ • Batch Size: Samples processed at once    │"
    echo -e "│ • Epoch: Complete pass through dataset     │"
    echo -e "│ • TFLOPS: Processing speed (higher=faster) │"
    echo -e "└────────────────────────────────────────────┘"
}

show_dataset_preview() {
    local dataset_type=$1
    echo -e "\n${BLUE}📊 Dataset Overview:${NC}"
    echo -e "┌────────────────────────────────────┐"
    case $dataset_type in
        "synthetic")
            echo -e "│ Type: Synthetic Training Data      │"
            echo -e "│ Size: 10,000 samples              │"
            echo -e "│ Split: 80% train, 20% validation  │"
            ;;
        "huggingface")
            echo -e "│ Source: Hugging Face Hub          │"
            python -c "
from datasets import load_dataset
dataset = load_dataset('$dataset_name', split='train')
print(f'│ Size: {len(dataset):,} samples')
print(f'│ Features: {list(dataset.features.keys())}')
            "
            ;;
        "custom")
            echo -e "│ Source: Local Dataset             │"
            echo -e "│ Path: $dataset_path               │"
            ;;
    esac
    echo -e "└────────────────────────────────────┘"
}

show_performance_estimates() {
    echo -e "\n${BLUE}⚡️ Performance Analysis:${NC}"
    python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig()
flops = config.total_tflops

print(f'┌────────────────────────────────────┐')
print(f'│ Hardware Configuration:            │')
print(f'│ • Devices: {config.num_devices:<21} │')
print(f'│ • Total Memory: {config.total_memory_gb:.1f}GB           │')
print(f'│ • Computing Power: {flops:.1f} TFLOPS     │')
print(f'└────────────────────────────────────┘')

print(f'\n{BLUE}⏱️  Estimated Training Times:{NC}')
print(f'┌────────────────────────────────────┐')
print(f'│ Small Model:  {0.5 * 3600 / flops:.1f} hours          │')
print(f'│ Medium Model: {2.0 * 3600 / flops:.1f} hours          │')
print(f'│ Large Model:  {7.0 * 3600 / flops:.1f} hours          │')
print(f'└────────────────────────────────────┘')
    "
}

show_model_architecture() {
    local model_type=$1
    echo -e "\n${BLUE}🧠 Model Architecture:${NC}"
    case $model_type in
        "transformer")
            echo -e "
┌─────────────────┐
│   Input Layer   │
└────────┬────────┘
         ▼
┌────────────────┐
│  Transformer   │
│     Block      │
│ ┌──────────┐   │
│ │Attention │   │
│ └────┬─────┘   │
│      ▼         │
│ ┌──────────┐   │
│ │  Dense   │   │
│ └──────────┘   │
└───────┬────────┘
        ▼
┌────────────────┐
│ Output Layer   │
└────────────────┘"
            ;;
        "mlp")
            echo -e "
┌─────────────┐
│ Input Layer │
└──────┬──────┘
       ▼
┌─────────────┐
│Dense + ReLU │
└──────┬──────┘
       ▼
┌─────────────┐
│Dense + ReLU │
└──────┬──────┘
       ▼
┌─────────────┐
│Output Layer │
└─────────────┘"
            ;;
    esac
}

# Setup environment
setup_environment() {
    echo -e "\n${BLUE}📦 Setting up environment...${NC}"
    
    # Check requirements first
    check_requirements
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv || {
            handle_error "env" "Failed to create virtual environment"
            exit 1
        }
    fi
    
    # Activate virtual environment
    source venv/bin/activate || {
        handle_error "env" "Failed to activate virtual environment"
        exit 1
    }
    
    # Install dependencies with error handling
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install --upgrade pip > /dev/null 2>&1 &
    spinner $!
    
    pip install -e ".[all]" > /dev/null 2>&1 || {
        handle_error "env" "Failed to install dependencies"
        exit 1
    }
    spinner $!
    
    echo -e "${GREEN}✓ Environment ready${NC}"
}

# Enhanced hardware detection with recommendations
detect_hardware() {
    echo -e "\n${BLUE}🔍 Analyzing your hardware configuration...${NC}"
    python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig()
devices = config.num_devices
memory = config.total_memory_gb

print(f'\n🖥️  Detected {devices} Apple Silicon device(s)')
print(f'💾 Total available memory: {memory:.1f}GB\n')

if devices > 1:
    print('✨ Great! Multi-device training enabled.')
    print('   Your training will automatically use all devices.')
else:
    print('💡 Training will use single device mode.')

if memory >= 16:
    print('🚀 Excellent memory capacity for large models!')
else:
    print('💡 Memory optimization enabled for efficient training.')
    "
}

# Guide users through distributed setup
setup_distributed() {
    echo -e "\n${BLUE}🌐 Distributed Training Setup${NC}"
    
    # Check if multiple devices are available locally
    local_devices=$(python -c "
from mlx_train.core.hardware import HardwareConfig
print(HardwareConfig().num_devices)
    ")
    
    echo -e "\n${YELLOW}Available Computing Power:${NC}"
    python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig()
total_flops = config.total_tflops

print(f'📊 Current Device(s): {config.num_devices}')
print(f'💪 Total Power: {total_flops:.2f} TFLOPS')
print(f'💾 Total Memory: {config.total_memory_gb:.1f}GB')
    "
    
    echo -e "\n${YELLOW}Would you like to add more Apple Silicon devices?${NC}"
    echo -e "1) Yes, connect via Network (WiFi/Ethernet)"
    echo -e "2) Yes, connect via Thunderbolt"
    echo -e "3) No, continue with current device(s)"
    read -p "Enter choice [1-3]: " connect_choice
    
    case $connect_choice in
        1)
            echo -e "\n${BLUE}📡 Network Connection Guide:${NC}"
            echo -e "1. Ensure all devices are on the same network"
            echo -e "2. Note down the IP addresses of each device"
            echo -e "3. We'll help you configure MPI for distributed training"
            
            read -p "Ready to proceed? [Y/n]: " setup_network
            if [[ $setup_network =~ ^[Yy]$ ]]; then
                echo -e "\n${YELLOW}Running network setup...${NC}"
                mlx-train setup-distributed --network
                
                # Show new computing power after setup
                echo -e "\n${GREEN}✨ New Computing Power:${NC}"
                python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig(refresh=True)
print(f'🚀 Total Devices: {config.num_devices}')
print(f'💪 Total Power: {config.total_tflops:.2f} TFLOPS')
                "
            fi
            ;;
            
        2)
            echo -e "\n${BLUE}🔌 Thunderbolt Connection Guide:${NC}"
            echo -e "1. Connect devices via Thunderbolt cable"
            echo -e "2. Ensure devices are powered on"
            echo -e "3. We'll verify the connection"
            
            read -p "Ready to proceed? [Y/n]: " setup_thunderbolt
            if [[ $setup_thunderbolt =~ ^[Yy]$ ]]; then
                echo -e "\n${YELLOW}Checking Thunderbolt connections...${NC}"
                mlx-train setup-distributed --thunderbolt
                
                # Show new computing power
                echo -e "\n${GREEN}✨ New Computing Power:${NC}"
                python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig(refresh=True)
print(f'🚀 Total Devices: {config.num_devices}')
print(f'💪 Total Power: {config.total_tflops:.2f} TFLOPS')
                "
            fi
            ;;
            
        3)
            echo -e "${GREEN}Continuing with current device(s)${NC}"
            ;;
            
        *) echo -e "${RED}Invalid choice${NC}"; exit 1;;
    esac
    
    # Show training time estimates
    if [[ $model_type == "transformer" ]]; then
        echo -e "\n${BLUE}⏱️  Estimated Training Times:${NC}"
        python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig()
flops = config.total_tflops

# Rough estimates for different model sizes
small = 0.5 * 3600 / flops  # 0.5B params
medium = 2 * 3600 / flops   # 2B params
large = 7 * 3600 / flops    # 7B params

print(f'Small Model (0.5B):  {small:.1f} hours')
print(f'Medium Model (2B):   {medium:.1f} hours')
print(f'Large Model (7B):    {large:.1f} hours')
        "
    fi
}

# Add after setup_distributed() function:

discover_devices() {
    echo -e "\n${BLUE}🔍 Scanning for Apple Silicon devices...${NC}"
    
    # Get current device info
    local current_device=$(system_profiler SPHardwareDataType | grep "Model Name" | cut -d: -f2- | xargs)
    local current_memory=$(system_profiler SPHardwareDataType | grep "Memory:" | cut -d: -f2- | xargs)
    
    echo -e "\n${GREEN}Current Setup:${NC}"
    echo -e "┌─────────────────────┐"
    echo -e "│ $current_device     │ ← Active"
    echo -e "│ Memory: $current_memory   │"
    echo -e "│ Status: Connected   │"
    echo -e "└─────────────────────┘"
    
    # Scan network for other Apple Silicon devices
    echo -e "\n${YELLOW}Scanning network...${NC}"
    progress_bar 2 "Scanning"
    
    # Use arp to find devices and filter for Apple
    local discovered=$(arp -a | grep "(Apple)")
    if [ ! -z "$discovered" ]; then
        echo -e "\n${GREEN}Found Apple devices:${NC}"
        while IFS= read -r line; do
            local device_ip=$(echo $line | grep -oE "\([0-9.]+\)" | tr -d '()')
            echo -e "┌─────────────────────┐"
            echo -e "│ Device at: $device_ip│"
            echo -e "│ Status: Available   │"
            echo -e "└─────────────────────┘"
            echo -e "       ↑"
        done <<< "$discovered"
    else
        echo -e "\n${YELLOW}No additional Apple devices found${NC}"
    fi
}

# Enhanced model selection with descriptions
select_model() {
    echo -e "\n${BLUE}🧠 Choose Your Model Architecture${NC}"
    echo -e "\n${YELLOW}Available architectures:${NC}"
    echo -e "1) ${GREEN}Transformer${NC} (Recommended)"
    echo -e "   • Best for language tasks"
    echo -e "   • State-of-the-art architecture"
    echo -e "   • Efficient on Apple Silicon"
    echo
    echo -e "2) ${GREEN}MLP${NC}"
    echo -e "   • Simple and fast"
    echo -e "   • Good for basic tasks"
    echo -e "   • Excellent for learning"
    echo
    echo -e "3) ${GREEN}Custom Architecture${NC}"
    echo -e "   • Full flexibility"
    echo -e "   • Advanced users"
    echo -e "   • Maximum control"
    
    read -p "Enter your choice [1-3]: " model_choice
    
    case $model_choice in
        1) 
            model_type="transformer"
            echo -e "${GREEN}✓ Selected Transformer architecture${NC}"
            ;;
        2) 
            model_type="mlp"
            echo -e "${GREEN}✓ Selected MLP architecture${NC}"
            ;;
        3) 
            model_type="custom"
            echo -e "${GREEN}✓ Selected Custom architecture${NC}"
            ;;
        *) echo -e "${RED}Invalid choice${NC}"; exit 1;;
    esac
}

# Dataset selection
select_dataset() {
    echo -e "\n${BLUE}📚 Select dataset:${NC}"
    echo "1) Synthetic dataset (for testing)"
    echo "2) HuggingFace dataset"
    echo "3) Custom dataset"
    read -p "Enter choice [1-3]: " dataset_choice
    
    case $dataset_choice in
        1) dataset_type="synthetic";;
        2) 
            echo -e "\n${YELLOW}Enter HuggingFace dataset name (e.g., 'wikitext'):${NC}"
            read dataset_name
            dataset_type="huggingface"
            ;;
        3)
            echo -e "\n${YELLOW}Enter path to dataset:${NC}"
            read dataset_path
            dataset_type="custom"
            ;;
        *) echo -e "${RED}Invalid choice${NC}"; exit 1;;
    esac
}

# Training configuration
configure_training() {
    echo -e "\n${BLUE}⚙️  Configure training:${NC}"
    read -p "Batch size [default: 32]: " batch_size
    batch_size=${batch_size:-32}
    
    read -p "Number of epochs [default: 10]: " epochs
    epochs=${epochs:-10}
    
    read -p "Enable mixed precision? [Y/n]: " mixed_precision
    mixed_precision=${mixed_precision:-Y}
}

# Error handling functions
handle_error() {
    local error_type=$1
    local details=$2
    
    echo -e "\n${RED}⚠️ Oops! Something went wrong...${NC}"
    
    case $error_type in
        "network")
            echo -e "${YELLOW}Connection Issue Detected:${NC}"
            echo -e "• Unable to connect to other devices"
            echo -e "• Check that all devices are on the same network"
            echo -e "• Verify IP addresses are correct"
            echo -e "\n${BLUE}Would you like to:${NC}"
            echo -e "1) Retry connection"
            echo -e "2) Continue with current device only"
            echo -e "3) Show troubleshooting steps"
            read -p "Enter choice [1-3]: " recovery_choice
            
            case $recovery_choice in
                1) setup_distributed;;
                2) echo -e "${YELLOW}Continuing with single device...${NC}";;
                3) show_network_troubleshooting;;
                *) echo -e "${RED}Invalid choice. Continuing with current device.${NC}";;
            esac
            ;;
            
        "device_disconnect")
            echo -e "${YELLOW}Device Disconnection Detected:${NC}"
            echo -e "• A device was disconnected during training"
            echo -e "• Current progress has been saved"
            echo -e "\n${BLUE}Would you like to:${NC}"
            echo -e "1) Attempt to reconnect"
            echo -e "2) Continue with remaining devices"
            echo -e "3) Save checkpoint and exit"
            read -p "Enter choice [1-3]: " recovery_choice
            
            case $recovery_choice in
                1) 
                    echo -e "${YELLOW}Attempting to reconnect...${NC}"
                    setup_distributed
                    ;;
                2) 
                    echo -e "${YELLOW}Adjusting batch size and continuing...${NC}"
                    reconfigure_training
                    ;;
                3) 
                    save_checkpoint
                    exit 0
                    ;;
            esac
            ;;
            
        "memory")
            echo -e "${YELLOW}Memory Usage Warning:${NC}"
            echo -e "• System is running low on memory"
            echo -e "• Current usage: $details"
            echo -e "\n${BLUE}Recommended actions:${NC}"
            echo -e "1) Reduce batch size"
            echo -e "2) Enable memory optimization"
            echo -e "3) Save checkpoint and exit"
            read -p "Enter choice [1-3]: " recovery_choice
            
            case $recovery_choice in
                1) 
                    echo -e "${YELLOW}Reducing batch size...${NC}"
                    batch_size=$((batch_size / 2))
                    echo -e "${GREEN}New batch size: $batch_size${NC}"
                    ;;
                2) 
                    echo -e "${YELLOW}Enabling memory optimization...${NC}"
                    enable_memory_optimization
                    ;;
                3) 
                    save_checkpoint
                    exit 0
                    ;;
            esac
            ;;
            
        *)
            echo -e "${RED}An unexpected error occurred:${NC}"
            echo -e "$details"
            echo -e "\n${YELLOW}Please try:${NC}"
            echo -e "1. Checking your network connection"
            echo -e "2. Verifying all devices are powered on"
            echo -e "3. Ensuring sufficient memory is available"
            echo -e "\nFor help, visit: https://github.com/your-repo/issues"
            ;;
    esac
}

# Monitor system resources
monitor_resources() {
    while true; do
        # Check memory usage
        if [[ $(check_memory_usage) -gt 90 ]]; then
            handle_error "memory" "$(get_memory_stats)"
        fi
        
        # Check device connectivity
        if ! check_devices_connected; then
            handle_error "device_disconnect"
        fi
        
        sleep 5
    done
}

# Helper functions
check_memory_usage() {
    python -c "
import mlx.core as mx
if mx.metal.is_available():
    print(int(mx.metal.get_current_memory() / mx.metal.get_peak_memory() * 100))
else:
    print(0)
"
}

check_devices_connected() {
    python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig(refresh=True)
exit(0 if config.num_devices == $initial_devices else 1)
"
}

save_checkpoint() {
    echo -e "${BLUE}📦 Saving training checkpoint...${NC}"
    mlx-train save-checkpoint
    echo -e "${GREEN}✓ Checkpoint saved${NC}"
}

show_network_troubleshooting() {
    echo -e "\n${BLUE}🔧 Network Troubleshooting Steps:${NC}"
    echo -e "1. Verify all devices are on the same network"
    echo -e "2. Check firewall settings"
    echo -e "3. Try these commands on each device:"
    echo -e "   $ ping [device-ip]"
    echo -e "   $ ssh [device-name]"
    echo -e "4. Ensure MPI is installed: mpirun --version"
}

# Step tracking
TOTAL_STEPS=8
show_step() {
    local step=$1
    local description=$2
    echo -e "\n${BLUE}Step ${step}/${TOTAL_STEPS}: ${GREEN}${description}${NC}"
    echo -e "${YELLOW}$(printf '─%.0s' {1..50})${NC}"
}

# Training visualization
show_training_status() {
    local epoch=$1
    local total_epochs=$2
    local loss=$3
    local throughput=$4
    
    # Clear previous status
    echo -ne "\033[K"
    
    # Show epoch progress
    local width=30
    local progress=$((epoch * width / total_epochs))
    echo -ne "\r${BLUE}Training Progress: ${NC}["
    for ((i=0; i<width; i++)); do
        if [ $i -lt $progress ]; then
            echo -ne "${GREEN}=${NC}"
        else
            echo -ne "-"
        fi
    done
    echo -ne "] ${GREEN}${epoch}/${total_epochs}${NC}"
    
    # Show metrics
    echo -e "\n${BLUE}Metrics:${NC}"
    echo -e "• Loss: ${GREEN}${loss}${NC}"
    echo -e "• Throughput: ${GREEN}${throughput} samples/sec${NC}"
}

# Device utilization visualization
show_device_status() {
    echo -e "\n${BLUE}Device Utilization:${NC}"
    python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig()
for i in range(config.num_devices):
    memory = mx.metal.get_current_memory() / 1e9 if mx.metal.is_available() else 0
    total = config.total_memory_gb / config.num_devices
    pct = int(memory / total * 100)
    bar = '=' * (pct // 2) + ' ' * (50 - pct // 2)
    print(f'Device {i}: [{bar}] {pct}% ({memory:.1f}GB)')
    "
}

# Enhanced main flow with progress tracking
main() {
    show_welcome
    show_tooltips
    
    show_step 1 "Setting Up Environment"
    setup_environment
    
    show_step 2 "Detecting Hardware"
    detect_hardware
    setup_distributed
    show_performance_estimates
    
    show_step 3 "Building Model"
    select_model
    show_model_architecture $model_type
    
    show_step 4 "Preparing Dataset"
    select_dataset
    show_dataset_preview $dataset_type
    
    show_step 5 "Configuring Training"
    configure_training
    
    show_step 6 "Training Model"
    echo -e "\n${BLUE}🚀 Starting training...${NC}"
    start_training
    
    show_step 7 "Exporting Model"
    export_model
    
    show_step 8 "Deploying Model"
    echo -e "\n${YELLOW}Would you like to run inference with your trained model?${NC}"
    echo -e "1) Start local API server"
    echo -e "2) Run CLI inference"
    echo -e "3) Start streaming server"
    echo -e "4) Skip deployment"
    read -p "Enter choice [1-4]: " deploy_choice
    
    case $deploy_choice in
        1|2|3)
            echo -e "\n${BLUE}🌐 Starting model server...${NC}"
            serve_model $deploy_choice
            
            # Show example usage
            echo -e "\n${GREEN}✨ Model is ready for inference!${NC}"
            if [ $deploy_choice -eq 1 ]; then
                echo -e "${YELLOW}API endpoint: ${NC}http://localhost:8000/v1/generate"
                echo -e "${YELLOW}Example request:${NC}"
                echo 'curl -X POST http://localhost:8000/v1/generate \'
                echo '  -H "Content-Type: application/json" \'
                echo '  -d '"'"'{"prompt": "Hello, how are you?"}'"'"
            elif [ $deploy_choice -eq 2 ]; then
                echo -e "${YELLOW}Enter prompts directly in the CLI${NC}"
            else
                echo -e "${YELLOW}Streaming endpoint: ${NC}http://localhost:8000/v1/generate"
                echo -e "${YELLOW}Connect with WebSocket for real-time responses${NC}"
            fi
            ;;
        4)
            echo -e "\n${BLUE}Skipping deployment${NC}"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
    
    echo -e "\n${GREEN}✨ Your AI journey is complete!${NC}"
    echo -e "${BLUE}📊 View training results: ./experiments${NC}"
    echo -e "${BLUE}💾 Find your model: ./exports${NC}"
    if [ $deploy_choice != 4 ]; then
        echo -e "${BLUE}🌐 Model is serving at: http://localhost:8000${NC}"
    fi
}

# Execute main function
main 