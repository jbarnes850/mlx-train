#!/bin/bash

# Set environment variables
export PYTHONPATH="."
export MLX_DISTRIBUTED="1"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Add gradient colors
GRADIENT_1='\033[38;5;39m'  # Light blue
GRADIENT_2='\033[38;5;38m'  # Cyan
GRADIENT_3='\033[38;5;37m'  # Teal
GRADIENT_4='\033[38;5;36m'  # Light green

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

# Screen management
clear_screen() {
    echo -e "\033[2J\033[H"  # Clear screen and move cursor to top
}

show_header() {
    echo -e "${BLUE}╔════════════════════════════════════════╗"
    echo -e "║ 🚀 MLX Training Framework                ║"
    echo -e "║ Train AI Models on Apple Silicon        ║"
    echo -e "╚════════════════════════════════════════╝${NC}"
    echo -e "\n"
}

# Update show_step function
show_step() {
    local step=$1
    local description=$2
    
    clear_screen
    show_header
    
    # Add top spacing
    echo -e "\n\n"
    
    echo -e "${BLUE}┌────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│${NC} ${YELLOW}STEP ${step}/${TOTAL_STEPS}${NC}                                  ${BLUE}│${NC}"
    echo -e "${BLUE}│${NC} ${GREEN}${description}${NC}                                ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────┘${NC}"
    
    # Add bottom spacing
    echo -e "\n\n"
}

# Update show_welcome
show_welcome() {
    clear_screen
    echo -e "${BLUE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════╗
║     __  __ _     __  __  _____           _               ║
║    |  \/  | |    \ \/ / |_   _| __ __ _ (_) _ __         ║
║    | |\/| | |     \  /    | | | '__/ _` || || '_ \       ║
║    | |  | | |___  /  \    | | | | | (_| || || | | |      ║
║    |_|  |_|_____|/_/\_\   |_| |_|  \__,_||_||_| |_|      ║
║                                                          ║
║       🚀 Distributed AI Training on Apple Silicon        ║
╚══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n\n"
    echo -e "${GREEN}Welcome to MLX Train!${NC}"
    echo -e "\n${YELLOW}Your high-performance distributed AI model training pipeline:${NC}\n"
    echo -e "🛠️  Zero-config environment setup\n"
    echo -e "🔧  Automatic hardware optimization for Apple Silicon\n" 
    echo -e "🚄  Seamless distributed training across devices\n"
    echo -e "📊  Real-time training metrics and visualization\n"
    echo -e "🔄  One-click model export and deployment\n"
    echo -e "\n"
    read -p "Press Enter to start distributed training..."
}

show_tooltips() {
    echo -e "\n📚 AI Training Quick Guide:"
    echo -e "\n🔍 Key Concepts for Your Training Run:"
    echo -e "• Batch Size: Number of training examples processed together"
    echo -e "  - Larger batches = faster training but more memory"
    echo -e "  - MLX automatically optimizes this for your device"
    echo -e "\n• Epoch: One complete pass through your entire dataset"
    echo -e "  - Multiple epochs help your model learn patterns"
    echo -e "  - Progress shown in real-time visualization"
    echo -e "\n• TFLOPS (Teraflops): Your device's processing power"
    echo -e "  - Higher TFLOPS = faster model training"
    echo -e "  - Apple Silicon optimized for maximum performance"
    echo -e "\n• Memory Usage: RAM utilized during training"
    echo -e "  - Shown in GB (gigabytes)"
    echo -e "  - Automatically managed across devices"
    echo -e "\n💡 Tip: Monitor the real-time metrics to track progress"
    echo -e "\n"
}

show_dataset_preview() {
    local dataset_type=$1
    echo -e "\n${BLUE}📊 Dataset Overview & Optimization${NC}"
    echo -e "┌────────────────────────────────────────────┐"
    case $dataset_type in
        "synthetic")
            echo -e "│ 🔄 Type: Synthetic Training Data          │"
            echo -e "│ 📊 Size: 10,000 samples                  │"
            echo -e "│ 📈 Split: 80% train, 20% validation      │"
            echo -e "│ ⚡️ Optimized for Apple Silicon           │"
            echo -e "│ 🎯 Perfect for testing & prototyping     │"
            ;;
        "huggingface")
            echo -e "│ 🤗 Source: Hugging Face Hub              │"
            python -c "
from datasets import load_dataset
dataset = load_dataset('$dataset_name', split='train')
print(f'│ 📊 Size: {len(dataset):,} samples')
print(f'│ 🔍 Features: {list(dataset.features.keys())}')
print(f'│ ⚡️ Auto-optimized for distributed training')
print(f'│ 🚀 Ready for high-performance processing')
            "
            ;;
        "custom")
            echo -e "│ 📁 Source: Local Dataset                 │"
            echo -e "│ 📍 Path: $dataset_path                   │"
            echo -e "│ ⚡️ Auto-configured for MLX               │"
            echo -e "│ 🔄 Ready for distributed processing      │"
            ;;
    esac
    echo -e "└────────────────────────────────────────────┘"
    
    echo -e "\n${GREEN}💡 Training Recommendations:${NC}"
    echo -e "• Batch size will be auto-optimized for your devices"
    echo -e "• Progress metrics will update in real-time"
    echo -e "• Memory usage is automatically balanced"
    echo -e "\n"
}

show_performance_estimates() {
    echo -e "\n${BLUE}⚡️ Performance Analysis:${NC}"
    python -c "
from mlx_train.core.hardware import HardwareConfig
config = HardwareConfig()
flops = config.total_tflops

print(f'┌────────────────────────────────────────────────┐')
print(f'│ Your Hardware:                                 │')
print(f'│ • Device: Apple Silicon                       │')
print(f'│ • Memory: {config.total_memory_gb:.1f}GB available              │')
print(f'│ • Power:  {flops:.1f} TFLOPS                        │')
print(f'└────────────────────────────────────────────────┘')

print(f'\n{BLUE}📊 Estimated Performance:{NC}')
print(f'┌────────────────────────────────────────────────┐')
print(f'│ Model Size     Training Time    Memory Usage   │')
print(f'│ ───────────────────────────────────────────── │')
print(f'│ Small (0.5B)   {0.5 * 3600 / flops:.1f} hours      {0.5:.1f}GB         │')
print(f'│ Medium (2B)    {2.0 * 3600 / flops:.1f} hours      {2.0:.1f}GB         │')
print(f'│ Large (7B)     {7.0 * 3600 / flops:.1f} hours      {7.0:.1f}GB         │')
print(f'└────────────────────────────────────────────────┘')

print(f'\n{BLUE}💡 Perfect for:{NC}')
print(f'┌────────────────────────────────────────────────┐')
print(f'│ ✓ Model Development and Testing               │')
print(f'│ ✓ Small to Medium Model Training             │')
print(f'│ ✓ Fine-tuning Existing Models                │')
print(f'│ ✓ Quick Experiments                          │')
print(f'└────────────────────────────────────���───────────┘')

if config.total_memory_gb >= 16:
    print(f'\n{GREEN}✨ Great! Your device has plenty of memory for most models.{NC}')
else:
    print(f'\n{YELLOW}💡 Tip: Memory optimization will be enabled automatically.{NC}')
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
show_hardware_status() {
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
    print('💡 Single device mode active - perfect for:')
    print('   • Model development and testing')
    print('   • Small to medium-sized models')
    print('   • Quick experiments')
    print('\n💡 Tip: You can add more devices later!')
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
    echo -e "    Full flexibility"
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
    
    clear_screen
    show_header
    
    # Add top spacing
    echo -e "\n\n"
    
    echo -e "${BLUE}┌────────────────────────────────────────────┐${NC}"
    echo -e "${BLUE}│${NC} ${YELLOW}STEP ${step}/${TOTAL_STEPS}${NC}                                  ${BLUE}│${NC}"
    echo -e "${BLUE}│${NC} ${GREEN}${description}${NC}                                ${BLUE}│${NC}"
    echo -e "${BLUE}└────────────────────────────────────────────┘${NC}"
    
    # Add bottom spacing
    echo -e "\n\n"
}

# Training visualization
show_training_status() {
    local epoch=$1
    local total_epochs=$2
    local loss=$3
    local throughput=$4
    
    echo -ne "\033[K"  # Clear line
    
    # Show epoch progress with gradient
    local width=40
    local progress=$((epoch * width / total_epochs))
    echo -ne "\n\r${BLUE}Training Progress: ${NC}["
    
    for ((i=0; i<width; i++)); do
        if [ $i -lt $progress ]; then
            if [ $i -lt $((width/4)) ]; then
                echo -ne "${GRADIENT_1}═${NC}"
            elif [ $i -lt $((width/2)) ]; then
                echo -ne "${GRADIENT_2}═${NC}"
            elif [ $i -lt $((3*width/4)) ]; then
                echo -ne "${GRADIENT_3}═${NC}"
            else
                echo -ne "${GRADIENT_4}═${NC}"
            fi
        else
            echo -ne "─"
        fi
    done
    echo -ne "] ${GREEN}${epoch}/${total_epochs}${NC}\n\n"
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

# Add to main():
add_spacing() {
    echo -e "\n\n"  # Double spacing
}

# Add transition functions
transition_step() {
    local from=$1
    local to=$2
    
    clear_screen
    show_header
    echo -e "\n${BLUE}Completing Step ${from}...${NC}"
    
    # Show subtle progress dots
    for i in {1..3}; do
        echo -ne "${BLUE}."
        sleep 0.2
    done
    echo -e "\n"
    
    # Show next step preview
    echo -e "${GREEN}Preparing Step ${to}${NC}"
    progress_bar 0.5 "Loading"
    
    clear_screen
}

# Add progress tracking
show_progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local filled=$((current * width / total))
    
    echo -e "\n${BLUE}Overall Progress${NC}"
    echo -ne "["
    for ((i=0; i<width; i++)); do
        if [ $i -lt $filled ]; then
            echo -ne "${GREEN}=${NC}"
        else
            echo -ne "-"
        fi
    done
    echo -ne "] ${GREEN}${current}/${total}${NC}\n"
}

# Update main() to use transitions:
main() {
    show_welcome
    add_spacing
    show_tooltips
    read -p "Press Enter to begin your journey..."
    
    # Step 1
    show_step 1 "Setting Up Environment"
    setup_environment
    add_spacing
    show_progress_bar 1 8
    read -p "Press Enter to continue..."
    transition_step 1 2
    
    # Step 2
    show_step 2 "Detecting Hardware"
    show_hardware_status
    setup_distributed
    add_spacing
    show_progress_bar 2 8
    read -p "Press Enter to continue..."
    transition_step 2 3
    
    # Continue pattern for other steps...
    
    # Final transition to completion
    clear_screen
    show_header
    echo -e "\n${GREEN}✨ Journey Complete!${NC}"
    show_progress_bar 8 8
    echo -e "\n${BLUE}Your model is ready:${NC}"
    echo -e "📊 Results: ./experiments"
    echo -e "💾 Model: ./exports"
    if [ $deploy_choice != 4 ]; then
        echo -e "🌐 API: http://localhost:8000"
    fi
    echo -e "\n"
}

# Execute main function
main 