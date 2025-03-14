#!/bin/bash
# =============================================================================
# GaiaNet Multi-Instance Setup (Improved)
# =============================================================================

# Colors for output
GREEN="\033[0;32m"
RESET="\033[0m"

# -------------------------------
# System-Wide Dependencies & Checks
# -------------------------------

echo "📦 Installing system dependencies..."
sudo apt update -y && sudo apt install -y pciutils libgomp1 curl wget build-essential libglvnd-dev pkg-config libopenblas-dev libomp-dev
sudo apt upgrade -y && sudo apt update

# Detect if running inside WSL
IS_WSL=false
if grep -qi microsoft /proc/version; then
    IS_WSL=true
    echo "🖥️ Running inside WSL."
else
    echo "🖥️ Running on a native Ubuntu system."
fi

# Check for NVIDIA GPU presence
check_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null || lspci | grep -i nvidia &> /dev/null; then
        echo "✅ NVIDIA GPU detected."
        return 0
    else
        echo "⚠️ No NVIDIA GPU found."
        return 1
    fi
}

# Determine system type: VPS, Laptop, or Desktop
check_system_type() {
    vps_type=$(systemd-detect-virt)
    if echo "$vps_type" | grep -qiE "kvm|qemu|vmware|xen|lxc"; then
        echo "✅ This is a VPS."
        return 0  # VPS
    elif ls /sys/class/power_supply/ 2>/dev/null | grep -q "^BAT[0-9]"; then
        echo "✅ This is a Laptop."
        return 1  # Laptop
    else
        echo "✅ This is a Desktop."
        return 2  # Desktop
    fi
}

# Check if CUDA is installed (by detecting nvcc)
check_cuda_installed() {
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K\d+\.\d+' | cut -d. -f1)
        echo "✅ CUDA version $CUDA_VERSION is already installed."
        return 0
    else
        echo "⚠️ CUDA is not installed."
        return 1
    fi
}

# Set up CUDA environment variables (system-wide)
setup_cuda_env() {
    echo "🔧 Setting up CUDA environment variables..."
    sudo tee /etc/profile.d/cuda.sh > /dev/null <<EOF
export PATH=/usr/local/cuda-12.8/bin\${PATH:+:\${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
EOF
    source /etc/profile.d/cuda.sh
}

# -------------------------------
# CUDA Toolkit Installation
# -------------------------------

install_cuda() {
    if $IS_WSL; then
        echo "🖥️ Installing CUDA for WSL 2..."
        PIN_FILE="cuda-wsl-ubuntu.pin"
        PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin"
        DEB_FILE="cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb"
        DEB_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/${DEB_FILE}"
    else
        echo "🖥️ Installing CUDA for Ubuntu 24.04..."
        PIN_FILE="cuda-ubuntu2404.pin"
        PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin"
        DEB_FILE="cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb"
        DEB_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/${DEB_FILE}"
    fi

    echo "📥 Downloading $PIN_FILE from $PIN_URL..."
    wget "$PIN_URL" || { echo "❌ Failed to download $PIN_FILE"; exit 1; }

    sudo mv "$PIN_FILE" /etc/apt/preferences.d/cuda-repository-pin-600 \
        || { echo "❌ Failed to move $PIN_FILE"; exit 1; }

    # Remove any existing DEB file and download a fresh copy
    [ -f "$DEB_FILE" ] && { echo "🗑️ Removing existing $DEB_FILE..."; rm -f "$DEB_FILE"; }
    echo "📥 Downloading $DEB_FILE from $DEB_URL..."
    wget "$DEB_URL" || { echo "❌ Failed to download $DEB_FILE"; exit 1; }

    sudo dpkg -i "$DEB_FILE" || { echo "❌ Failed to install $DEB_FILE"; exit 1; }
    sudo cp /var/cuda-repo-*/cuda-*-keyring.gpg /usr/share/keyrings/ \
        || { echo "❌ Failed to copy CUDA keyring"; exit 1; }

    echo "🔄 Updating package list..."
    sudo apt-get update || { echo "❌ Failed to update package list"; exit 1; }
    echo "🔧 Installing CUDA Toolkit 12.8..."
    sudo apt-get install -y cuda-toolkit-12-8 || { echo "❌ Failed to install CUDA Toolkit 12.8"; exit 1; }

    echo "✅ CUDA Toolkit 12.8 installed successfully."
    setup_cuda_env
}

# -------------------------------
# GaiaNet Installation (Per Instance)
# -------------------------------

# Install GaiaNet into a given directory.
install_gaianet() {
    local install_dir="$1"
    echo "🔧 Installing GaiaNet in ${install_dir}..."
    # Download installer script into the target directory.
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep 'release' | awk '{print $6}' | cut -d',' -f1 | sed 's/V//g' | cut -d'.' -f1)
        echo "✅ CUDA version detected: $CUDA_VERSION"
        if [[ "$CUDA_VERSION" == "11" || "$CUDA_VERSION" == "12" ]]; then
            echo "🔧 Installing GaiaNet with ggmlcuda $CUDA_VERSION..."
            # Using 0.4.21 for GPU-supported installation.
            curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/download/0.4.21/install.sh' -o "${install_dir}/install.sh" \
                || { echo "❌ Failed to download install.sh"; exit 1; }
            chmod +x "${install_dir}/install.sh"
            (cd "$install_dir" && ./install.sh --ggmlcuda "$CUDA_VERSION" --base "$install_dir") \
                || { echo "❌ GaiaNet installation with CUDA failed."; exit 1; }
            return
        fi
    fi
    echo "⚠️ Installing GaiaNet without GPU support..."
    curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/download/0.4.20/install.sh' -o "${install_dir}/install.sh" \
        || { echo "❌ Failed to download install.sh"; exit 1; }
    chmod +x "${install_dir}/install.sh"
    (cd "$install_dir" && ./install.sh --base "$install_dir") \
        || { echo "❌ GaiaNet installation without GPU support failed."; exit 1; }
}

# Add the instance's bin directory to PATH in ~/.bashrc
add_gaianet_to_path() {
    local install_dir="$1"
    # Check if already added to avoid duplicate entries.
    if ! grep -q "${install_dir}/bin" ~/.bashrc; then
        echo "export PATH=${install_dir}/bin:\$PATH" >> ~/.bashrc
        echo "✅ Added ${install_dir}/bin to PATH."
    fi
    # Update current session
    export PATH="${install_dir}/bin:$PATH"
}

# -------------------------------
# Main Script Execution
# -------------------------------

# If NVIDIA GPU is present and CUDA is not yet set up, handle CUDA installation.
if check_nvidia_gpu; then
    if ! setup_cuda_env; then
        echo "⚠️ CUDA environment not set up. Checking CUDA installation..."
        if check_cuda_installed; then
            echo "⚠️ CUDA is installed but not set up correctly. Please fix the CUDA environment."
        else
            echo "CUDA is not installed. Installing CUDA..."
            install_cuda || { echo "❌ Failed to install CUDA. Exiting."; exit 1; }
            setup_cuda_env
        fi
    else
        echo "✅ CUDA environment is already set up."
    fi
fi

# Prompt for the number of instances (1-4)
read -p "How many GaiaNet instances do you want to install? (1-4): " NUM_INSTANCES
if ! [[ "$NUM_INSTANCES" =~ ^[1-4]$ ]]; then
    echo "❌ Please enter a valid number between 1 and 4."
    exit 1
fi

# Loop through each instance installation
for (( i=1; i<=NUM_INSTANCES; i++ )); do
    echo "==========================================================="
    echo -e "${GREEN}Setting up GaiaNet instance $i...${RESET}"
    
    # Define unique installation directory and port for this instance.
    GAIA_DIR="$HOME/gaianet$i"
    GAIA_PORT="909$i"
    mkdir -p "$GAIA_DIR" || { echo "❌ Failed to create directory $GAIA_DIR"; exit 1; }
    
    # Install GaiaNet into this directory.
    install_gaianet "$GAIA_DIR"
    
    # Verify installation and add to PATH.
    if [ -f "$GAIA_DIR/bin/gaianet" ]; then
        echo "✅ GaiaNet installed successfully in $GAIA_DIR."
        add_gaianet_to_path "$GAIA_DIR"
    else
        echo "❌ GaiaNet installation failed in $GAIA_DIR. Exiting."
        exit 1
    fi

    # Determine system type and select configuration URL.
    check_system_type
    SYSTEM_TYPE=$?  # 0: VPS, 1: Laptop, 2: Desktop
    if [[ $SYSTEM_TYPE -eq 0 ]]; then
        CONFIG_URL="https://raw.githubusercontent.com/abhiag/Gaia_Node/main/config2.json"
    elif [[ $SYSTEM_TYPE -eq 1 ]]; then
        if ! check_nvidia_gpu; then
            CONFIG_URL="https://raw.githubusercontent.com/abhiag/Gaia_Node/main/config2.json"
        else
            CONFIG_URL="https://raw.githubusercontent.com/abhiag/Gaia_Node/main/config1.json"
        fi
    elif [[ $SYSTEM_TYPE -eq 2 ]]; then
        if ! check_nvidia_gpu; then
            CONFIG_URL="https://raw.githubusercontent.com/abhiag/Gaia_Node/main/config2.json"
        else
            CONFIG_URL="https://raw.githubusercontent.com/abhiag/Gaia_Node/main/config3.json"
        fi
    fi

    # Initialize GaiaNet for this instance.
    echo "⚙️ Initializing GaiaNet in ${GAIA_DIR}..."
    "$GAIA_DIR/bin/gaianet" init --config "$CONFIG_URL" --base "$GAIA_DIR" || { echo "❌ GaiaNet initialization failed in ${GAIA_DIR}!"; exit 1; }
    
    # Configure and start the node.
    echo "🚀 Starting GaiaNet node in ${GAIA_DIR} on port ${GAIA_PORT}..."
    "$GAIA_DIR/bin/gaianet" config --base "$GAIA_DIR" --port "$GAIA_PORT" --domain gaia.domains
    "$GAIA_DIR/bin/gaianet" start --base "$GAIA_DIR" || { echo "❌ Failed to start GaiaNet node in ${GAIA_DIR}!"; exit 1; }
    
    echo "🔍 Fetching GaiaNet node information in ${GAIA_DIR}..."
    "$GAIA_DIR/bin/gaianet" info --base "$GAIA_DIR" || { echo "❌ Failed to fetch node info in ${GAIA_DIR}!"; exit 1; }
done

# Final message
echo "==========================================================="
echo -e "${GREEN}🎉 Congratulations! All GaiaNet node installations have been set up successfully!${RESET}"
echo "==========================================================="
