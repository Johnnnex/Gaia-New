#!/bin/bash

# ===============================
#   GaiaNet Multi-Instance Setup
# ===============================

# Colors for output
GREEN="\033[0;32m"
RESET="\033[0m"

# --- System-Wide Dependencies ---
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

# -------------------------------
# Functions (System Checks & Setup)
# -------------------------------

check_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null || lspci | grep -i nvidia &> /dev/null; then
        echo "✅ NVIDIA GPU detected."
        return 0
    else
        echo "⚠️ No NVIDIA GPU found."
        return 1
    fi
}

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

install_cuda() {
    if $IS_WSL; then
        echo "🖥️ Installing CUDA for WSL 2..."
        PIN_FILE="cuda-wsl-ubuntu.pin"
        PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin"
        DEB_FILE="cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb"
        DEB_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb"
    else
        echo "🖥️ Installing CUDA for Ubuntu 24.04..."
        PIN_FILE="cuda-ubuntu2404.pin"
        PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin"
        DEB_FILE="cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb"
        DEB_URL="https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb"
    fi

    echo "📥 Downloading $PIN_FILE from $PIN_URL..."
    wget "$PIN_URL" || { echo "❌ Failed to download $PIN_FILE"; exit 1; }
    sudo mv "$PIN_FILE" /etc/apt/preferences.d/cuda-repository-pin-600 || { echo "❌ Failed to move $PIN_FILE"; exit 1; }

    if [ -f "$DEB_FILE" ]; then
        echo "🗑️ Deleting existing $DEB_FILE..."
        rm -f "$DEB_FILE"
    fi
    echo "📥 Downloading $DEB_FILE from $DEB_URL..."
    wget "$DEB_URL" || { echo "❌ Failed to download $DEB_FILE"; exit 1; }

    sudo dpkg -i "$DEB_FILE" || { echo "❌ Failed to install $DEB_FILE"; exit 1; }
    sudo cp /var/cuda-repo-*/cuda-*-keyring.gpg /usr/share/keyrings/ || { echo "❌ Failed to copy CUDA keyring"; exit 1; }

    echo "🔄 Updating package list..."
    sudo apt-get update || { echo "❌ Package list update failed"; exit 1; }
    echo "🔧 Installing CUDA Toolkit 12.8..."
    sudo apt-get install -y cuda-toolkit-12-8 || { echo "❌ CUDA Toolkit installation failed"; exit 1; }
    echo "✅ CUDA Toolkit 12.8 installed successfully."

    setup_cuda_env
}

setup_cuda_env() {
    echo "🔧 Setting up CUDA environment variables..."
    echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' | sudo tee /etc/profile.d/cuda.sh > /dev/null
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' | sudo tee -a /etc/profile.d/cuda.sh > /dev/null
    source /etc/profile.d/cuda.sh
}

# -------------------------------
# GaiaNet-Specific Functions (Per Instance)
# -------------------------------

# Modified install_gaianet: installs GaiaNet into a given directory.
install_gaianet() {
    local install_dir="$1"
    echo "🔧 Installing GaiaNet in ${install_dir}..."
    # Download the installer script into the target directory.
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep 'release' | awk '{print $6}' | cut -d',' -f1 | sed 's/V//g' | cut -d'.' -f1)
        echo "✅ CUDA version detected: $CUDA_VERSION"
        if [[ "$CUDA_VERSION" == "11" || "$CUDA_VERSION" == "12" ]]; then
            echo "🔧 Installing GaiaNet with ggmlcuda $CUDA_VERSION..."
            curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/download/0.4.20/install.sh' -o "${install_dir}/install.sh" \
                || { echo "❌ Failed to download install.sh"; exit 1; }
            chmod +x "${install_dir}/install.sh"
            # Change into the installation directory and pass the --base flag so that installation occurs in ${install_dir}
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
        || { echo "❌ GaiaNet installation without GPU failed."; exit 1; }
}


# Modified add_gaianet_to_path: adds the instance's bin directory to ~/.bashrc.
add_gaianet_to_path() {
    local install_dir="$1"
    echo "export PATH=${install_dir}/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
}

# -------------------------------
# Main Script Execution
# -------------------------------

# If an NVIDIA GPU is present and CUDA is not yet installed, install CUDA (system-wide).
if check_nvidia_gpu; then
    setup_cuda_env
    # Install CUDA only once (first instance will trigger this)
    # install_cuda
    setup_cuda_env
fi

# Prompt the user for how many instances (directories) to set up (max 4)
read -p "How many GaiaNet instances do you want to install? (1-4): " NUM_INSTANCES
if ! [[ "$NUM_INSTANCES" =~ ^[1-4]$ ]]; then
    echo "❌ Please enter a valid number between 1 and 4."
    exit 1
fi

# Loop through each instance installation
for (( i=1; i<=NUM_INSTANCES; i++ )); do
    echo "==========================================================="
    echo -e "${GREEN}Setting up GaiaNet instance $i...${RESET}"
    
    # Define a unique installation directory for this instance.
    GAIA_DIR="$HOME/gaianet$i"
    GAIA_PORT="809$i"
    mkdir -p "$GAIA_DIR" || { echo "❌ Failed to create directory $GAIA_DIR"; exit 1; }
    
    # Install GaiaNet into this directory.
    # (CUDA is already installed system-wide; each instance uses the same CUDA support.)
    install_gaianet "$GAIA_DIR"
    
    # Verify installation
    if [ -f "$GAIA_DIR/bin/gaianet" ]; then
        echo "✅ GaiaNet installed successfully in $GAIA_DIR."
        add_gaianet_to_path "$GAIA_DIR"
    else
        echo "❌ GaiaNet installation failed in $GAIA_DIR. Exiting."
        exit 1
    fi

    # Determine system type and choose the appropriate config URL.
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

    # Initialize the GaiaNet node using the configuration.
    echo "⚙️ Initializing GaiaNet in ${GAIA_DIR}..."
    "$GAIA_DIR/bin/gaianet" init --config "$CONFIG_URL" --base "$GAIA_DIR" || { echo "❌ GaiaNet initialization failed in ${GAIA_DIR}!"; exit 1; }
    
    # Set the domain, start the node, and then show node info.
    echo "🚀 Starting GaiaNet node in ${GAIA_DIR}..."
    "$GAIA_DIR/bin/gaianet" config --base "$GAIA_DIR" --port "$GAIA_PORT" --domain gaia.domains
    "$GAIA_DIR/bin/gaianet" start --base "$GAIA_DIR" || { echo "❌ Error: Failed to start GaiaNet node in ${GAIA_DIR}!"; exit 1; }
    
    echo "🔍 Fetching GaiaNet node information in ${GAIA_DIR}..."
    "$GAIA_DIR/bin/gaianet" info --base "$GAIA_DIR" || { echo "❌ Error: Failed to fetch GaiaNet node information in ${GAIA_DIR}!"; exit 1; }
done

# Final message
echo "==========================================================="
echo -e "${GREEN}🎉 Congratulations! All GaiaNet node installations have been set up successfully!${RESET}"
echo "==========================================================="
