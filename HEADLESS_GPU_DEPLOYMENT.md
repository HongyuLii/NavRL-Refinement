#!/bin/bash
# ROS2 Deployment Guide for Headless GPU VMs (Lambda, GCP, AWS)
# This script demonstrates how to run the full NavRL ROS2 stack without a GUI

echo "============================================"
echo "NavRL ROS2 Headless GPU Deployment Guide"
echo "============================================"
echo ""
echo "This guide covers:"
echo "1. Building on a headless GPU VM"
echo "2. Running nodes with topic recording"
echo "3. Testing the full system without GUI"
echo ""

cat > /tmp/headless_deployment_guide.md << 'EOF'
# NavRL ROS2 Headless Deployment Guide

## Prerequisites
- Ubuntu 22.04 LTS
- ROS2 Humble installed
- NVIDIA GPU with CUDA
- Python 3.10

## Step 1: Prepare the GPU VM

### Install Isaac Go2 Simulator dependencies
```bash
# Create isaaclab environment (same as your local machine)
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install Isaac Lab (for Isaac Sim)
# Follow: https://github.com/Zhefan-Xu/isaac-go2-ros2
```

### Install NavRL dependencies
```bash
# Create NavRL environment
conda create -n NavRL python=3.10
conda activate NavRL
cd /path/to/NavRL/isaac-training
bash setup_deployment.sh
```

## Step 2: Build ROS2 Packages

```bash
# Create workspace
mkdir -p ~/navrl_ws/src
cd ~/navrl_ws/src
git clone https://github.com/HongyuLii/NavRL-Refinement.git
ln -s NavRL-Refinement/ros2 ros2_packages

# Build
cd ~/navrl_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## Step 3: Run on Headless VM (Terminal-based Testing)

### Terminal 1: Start ROS2 bag recording
```bash
# In a dedicated terminal, record all topics to a bag file
source /opt/ros/humble/setup.bash
source ~/navrl_ws/install/setup.bash
ros2 bag record -o /tmp/navrl_test -a
```

### Terminal 2: Launch Perception & Safety
```bash
source /opt/ros/humble/setup.bash
source ~/navrl_ws/install/setup.bash
ros2 launch navigation_runner perception.launch.py
```

### Terminal 3: Launch Navigation
```bash
conda activate NavRL
source /opt/ros/humble/setup.bash
source ~/navrl_ws/install/setup.bash
ros2 launch navigation_runner navigation.launch.py
```

### Terminal 4: Run Isaac Go2 Simulator
```bash
# Note: This MUST run on the GPU VM with display access (X11 or VNC)
# OR run in headless mode if using latest Isaac Sim container
conda activate isaaclab
cd /path/to/isaac-go2-ros2
python isaac-go2-ros2.py --headless  # If supported
```

## Step 4: Alternative - Run Everything in Docker (Recommended for Headless)

Create a Dockerfile:
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install ROS2 Humble
RUN apt-get update && apt-get install -y \
    curl gnupg lsb-release \
    && curl -sSL https://repo.ros2.org/ros.key | apt-key add - \
    && add-apt-repository "deb [arch=amd64,arm64] http://repo.ros2.org/ubuntu $(lsb_release -cs) main" \
    && apt-get install -y ros-humble-desktop

# Install dependencies
RUN apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip git

# Copy NavRL code
COPY NavRL-Refinement /home/NavRL
WORKDIR /home/NavRL/ros2

# Build
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

ENTRYPOINT ["/bin/bash"]
```

Build and run:
```bash
docker build -t navrl-ros2:latest .
docker run --gpus all --rm -it --network=host navrl-ros2:latest
```

## Step 5: Monitor without GUI

### Option 1: SSH with ROS2 CLI
```bash
# From your local machine
ssh user@gpu_vm

# List running nodes
ros2 node list

# Check topic bandwidth
ros2 topic bw /occupancy_map/voxel_map

# Check service availability
ros2 service list
ros2 service call /occupancy_map/raycast map_manager_msgs/RayCast "{...}"
```

### Option 2: ROS2 Bag Playback (Local Analysis)
```bash
# On your local machine
scp user@gpu_vm:/tmp/navrl_test ~/local_bag

# Analyze the bag
ros2 bag info ~/local_bag
ros2 bag play ~/local_bag  # Can visualize in RViz locally
```

### Option 3: Real-time Monitoring
```bash
# Forward topics over network
ssh -L localhost:11311:localhost:11311 user@gpu_vm

# On local machine, set ROS_DOMAIN_ID
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
ros2 node list  # See remote nodes
ros2 topic list  # See remote topics
```

## Troubleshooting on Headless GPU VM

### Issue 1: Isaac Sim needs display
**Solution:** Use X11 forwarding or VirtualGL
```bash
ssh -X user@gpu_vm  # Enable X11 forwarding
# OR
vglrun python isaac-go2-ros2.py  # Use VirtualGL
```

### Issue 2: Simulated sensor data not published
**Solution:** Check Isaac Sim output
```bash
ros2 topic list | grep -i camera
ros2 topic echo /camera/depth/image_raw --print-all-fields
```

### Issue 3: Navigation node waiting for service
**Solution:** Verify perception nodes are running
```bash
ros2 service list | grep occupancy_map
ros2 service call /occupancy_map/raycast ...
```

## Performance Tuning for GPU VMs

### Reduce CPU overhead
```bash
# Use ROS2 DDS tuning
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/path/to/cyclonedds.xml
```

### Monitor GPU usage
```bash
# In separate terminal
watch nvidia-smi
```

### Limit bandwidth
```bash
# If network is constrained
ros2 launch navigation_runner perception.launch.py \
  use_sim_time:=false \
  enable_gpu_acceleration:=true
```
EOF

cat /tmp/headless_deployment_guide.md
