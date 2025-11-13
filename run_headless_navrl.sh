#!/bin/bash
# run_headless_navrl.sh
# Complete automation script to run NavRL on a headless GPU VM
# Usage: bash run_headless_navrl.sh [--record-bag] [--isaac-headless]

set -e

RECORD_BAG=false
ISAAC_HEADLESS=false
NAMESPACE=$(date +%Y%m%d_%H%M%S)

while [[ $# -gt 0 ]]; do
    case $1 in
        --record-bag)
            RECORD_BAG=true
            shift
            ;;
        --isaac-headless)
            ISAAC_HEADLESS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "NavRL ROS2 Headless Launcher"
echo "=========================================="
echo "Recording bag: $RECORD_BAG"
echo "Isaac headless: $ISAAC_HEADLESS"
echo "Test namespace: $NAMESPACE"
echo ""

# Setup
source /opt/ros/humble/setup.bash
source ~/navrl_ws/install/setup.bash

# Create working directories
WORK_DIR="/tmp/navrl_$NAMESPACE"
BAG_DIR="$WORK_DIR/bags"
LOG_DIR="$WORK_DIR/logs"
mkdir -p $BAG_DIR $LOG_DIR

echo "[Setup] Working directory: $WORK_DIR"
echo "[Setup] Logging to: $LOG_DIR"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Shutting down NavRL ROS2..."
    echo "=========================================="
    
    # Kill all background processes
    jobs -p | xargs -r kill -9 2>/dev/null || true
    
    if [ "$RECORD_BAG" = true ]; then
        echo "Bag recording saved to: $BAG_DIR"
    fi
    
    echo "All processes stopped"
}

trap cleanup EXIT

# Start ROS2 bag recording (optional)
if [ "$RECORD_BAG" = true ]; then
    echo "[Bag] Starting ROS2 bag recording..."
    mkdir -p $BAG_DIR
    ros2 bag record -o $BAG_DIR/navrl_test -a > $LOG_DIR/bag.log 2>&1 &
    BAG_PID=$!
    echo "[Bag] Bag recorder started (PID: $BAG_PID)"
    sleep 2
fi

# Start Isaac Go2 Simulator
echo "[Isaac] Starting Isaac Go2 simulator..."
if [ "$ISAAC_HEADLESS" = true ]; then
    cd /path/to/isaac-go2-ros2
    conda run -n isaaclab python isaac-go2-ros2.py --headless > $LOG_DIR/isaac.log 2>&1 &
else
    cd /path/to/isaac-go2-ros2
    conda run -n isaaclab python isaac-go2-ros2.py > $LOG_DIR/isaac.log 2>&1 &
fi
ISAAC_PID=$!
echo "[Isaac] Simulator started (PID: $ISAAC_PID)"
echo "[Isaac] Waiting for simulator to initialize..."
sleep 10  # Give Isaac time to start

# Start perception module
echo "[Perception] Starting perception & safety modules..."
ros2 launch navigation_runner perception.launch.py > $LOG_DIR/perception.log 2>&1 &
PERCEPTION_PID=$!
echo "[Perception] Perception launched (PID: $PERCEPTION_PID)"
sleep 5

# Start safe action node
echo "[SafeAction] Starting safe action node..."
ros2 launch navigation_runner safe_action.launch.py > $LOG_DIR/safe_action.log 2>&1 &
SAFE_ACTION_PID=$!
echo "[SafeAction] Safe action node launched (PID: $SAFE_ACTION_PID)"
sleep 3

# Start navigation node
echo "[Navigation] Starting navigation node..."
conda run -n NavRL ros2 launch navigation_runner navigation.launch.py > $LOG_DIR/navigation.log 2>&1 &
NAV_PID=$!
echo "[Navigation] Navigation node launched (PID: $NAV_PID)"
sleep 2

# Print status
echo ""
echo "=========================================="
echo "NavRL ROS2 Stack is Running"
echo "=========================================="
echo "Isaac (PID: $ISAAC_PID)"
echo "Perception (PID: $PERCEPTION_PID)"
echo "SafeAction (PID: $SAFE_ACTION_PID)"
echo "Navigation (PID: $NAV_PID)"
echo ""
echo "Logs directory: $LOG_DIR"
echo "Bag directory: $BAG_DIR"
echo ""
echo "Commands to monitor (in another terminal):"
echo "  ros2 node list"
echo "  ros2 topic list"
echo "  ros2 service list"
echo "  tail -f $LOG_DIR/navigation.log"
echo ""
echo "Press Ctrl+C to stop all processes"
echo "=========================================="
echo ""

# Wait indefinitely
wait
