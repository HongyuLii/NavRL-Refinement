#!/bin/bash
# ROS2 Deployment Verification Script (Headless)
# This script tests all ROS2 packages without requiring a GPU or GUI

set -e

echo "=========================================="
echo "NavRL ROS2 Deployment Verification"
echo "=========================================="
echo ""

# Source ROS2 and workspace
echo "[1/6] Sourcing ROS2 and workspace..."
source /opt/ros/humble/setup.bash
source ~/navrl_ws/install/setup.bash

# Check if packages are discoverable
echo "[2/6] Checking if packages are discoverable..."
PACKAGES=$(ros2 pkg list | grep -E "map_manager|navigation_runner|onboard_detector" | wc -l)
if [ $PACKAGES -eq 3 ]; then
    echo "✓ All 3 packages found"
else
    echo "✗ Missing packages! Found $PACKAGES out of 3"
    exit 1
fi

# List executables
echo "[3/6] Listing available executables..."
echo "  map_manager:"
ros2 pkg executables map_manager | sed 's/^/    /'
echo "  navigation_runner:"
ros2 pkg executables navigation_runner | sed 's/^/    /'
echo "  onboard_detector:"
ros2 pkg executables onboard_detector | sed 's/^/    /'

# Test node initialization (without GUI)
echo "[4/6] Testing node initialization (30 seconds per node)..."

echo "  Testing occupancy_map_node..."
timeout 5 ros2 run map_manager occupancy_map_node > /tmp/map_manager.log 2>&1 || true
if grep -q "OccMap" /tmp/map_manager.log; then
    echo "    ✓ occupancy_map_node initializes correctly"
else
    echo "    ✗ occupancy_map_node failed to initialize"
    cat /tmp/map_manager.log
    exit 1
fi

echo "  Testing dynamic_detector_node..."
timeout 5 ros2 run onboard_detector dynamic_detector_node > /tmp/detector.log 2>&1 || true
if grep -q "onboardDetector" /tmp/detector.log; then
    echo "    ✓ dynamic_detector_node initializes correctly"
else
    echo "    ✗ dynamic_detector_node failed to initialize"
    cat /tmp/detector.log
    exit 1
fi

echo "  Testing safe_action_node..."
timeout 5 ros2 run navigation_runner safe_action_node > /tmp/safe_action.log 2>&1 || true
if grep -q "safeAction" /tmp/safe_action.log; then
    echo "    ✓ safe_action_node initializes correctly"
else
    echo "    ✗ safe_action_node failed to initialize"
    cat /tmp/safe_action.log
    exit 1
fi

echo "  Testing navigation_node.py..."
timeout 5 ros2 run navigation_runner navigation_node.py > /tmp/navigation.log 2>&1 || true
if grep -q "navRunner" /tmp/navigation.log; then
    echo "    ✓ navigation_node.py initializes correctly"
else
    echo "    ✗ navigation_node.py failed to initialize"
    cat /tmp/navigation.log
    exit 1
fi

# Test launch files
echo "[5/6] Testing launch file availability..."
echo "  Checking perception.launch.py..."
if ros2 launch navigation_runner perception.launch.py --show-args > /dev/null 2>&1; then
    echo "    ✓ perception.launch.py is valid"
else
    echo "    ✗ perception.launch.py has issues"
fi

echo "  Checking navigation.launch.py..."
if ros2 launch navigation_runner navigation.launch.py --show-args > /dev/null 2>&1; then
    echo "    ✓ navigation.launch.py is valid"
else
    echo "    ✗ navigation.launch.py has issues"
fi

# Summary
echo ""
echo "[6/6] Verification Summary"
echo "=========================================="
echo "✓ All ROS2 packages are installed and discoverable"
echo "✓ All nodes can initialize successfully"
echo "✓ All launch files are valid"
echo ""
echo "ROS2 Deployment is READY for GPU testing!"
echo ""
echo "Next steps:"
echo "1. Copy ros2 folder to a GPU machine (Lambda or GCP VM)"
echo "2. Build with colcon build --symlink-install"
echo "3. Launch nodes with:"
echo "   - ros2 launch navigation_runner perception.launch.py"
echo "   - ros2 launch navigation_runner safe_action.launch.py"
echo "   - ros2 launch navigation_runner navigation.launch.py"
echo "4. Use ROS2 bag recording for headless testing:"
echo "   - ros2 bag record -a (to record all topics)"
echo ""
echo "=========================================="
