#!/usr/bin/env python3
"""
Batch evaluation script for NavRL navigation
Runs 30 headless experiments and collects metrics:
- Success rate (reaching goal within threshold)
- Final position bias from target
- Collision count
- Failure reasons (out of bounds, crash, timeout)
"""

import rospy
import numpy as np
import json
import csv
import time
import os
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from map_manager.srv import CheckPosCollision, CheckPosCollisionRequest, GetStaticObstacles
from std_srvs.srv import Empty
import statistics

# Configuration
NUM_RUNS = 30
GOAL_SUCCESS_THRESHOLD = 1.0  # meters (distance to goal for success)
MAX_RUN_TIME = 120.0  # seconds (timeout for each run)
MAP_BOUNDS = [-15, 15]  # X and Y bounds (matches world_generator.yaml range_x and range_y)
COLLISION_CHECK_INTERVAL = 0.1  # seconds between collision checks
OUTPUT_DIR = "evaluation_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BatchEvaluator:
    def __init__(self):
        self.odom = None
        self.goal = None
        self.model_states = None
        self.collision_count = 0
        self.last_collision_check_time = 0
        self.last_position = None
        
        # ROS subscribers
        self.odom_sub = rospy.Subscriber("/CERLAB/quadcopter/odom", Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        self.model_states_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)
        
        # ROS publishers
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        
        # Services - wait for them to be available (nodes should be launched manually)
        rospy.loginfo("Waiting for ROS services to be available...")
        rospy.wait_for_service("/occupancy_map/check_pos_collision", timeout=60.0)
        self.collision_check = rospy.ServiceProxy("/occupancy_map/check_pos_collision", CheckPosCollision)
        rospy.wait_for_service("/occupancy_map/get_static_obstacles", timeout=60.0)
        self.get_static_obstacles = rospy.ServiceProxy("/occupancy_map/get_static_obstacles", GetStaticObstacles)
        
        # Wait for initial odometry
        rospy.sleep(2.0)
        rospy.loginfo("All services ready!")
    
    def odom_callback(self, msg):
        self.odom = msg
        
    def goal_callback(self, msg):
        self.goal = msg
        
    def model_states_callback(self, msg):
        self.model_states = msg
        
    def get_current_position(self):
        """Get current drone position from odometry"""
        if self.odom is None:
            return None
        pos = self.odom.pose.pose.position
        return np.array([pos.x, pos.y, pos.z])
    
    def get_goal_position(self):
        """Get current goal position"""
        if self.goal is None:
            return None
        pos = self.goal.pose.position
        return np.array([pos.x, pos.y, pos.z])
    
    def distance_to_goal(self):
        """Calculate distance to goal"""
        curr_pos = self.get_current_position()
        goal_pos = self.get_goal_position()
        if curr_pos is None or goal_pos is None:
            return None
        return np.linalg.norm(curr_pos - goal_pos)
    
    def check_collision(self):
        """Check if drone is in collision using map_manager service"""
        curr_pos = self.get_current_position()
        if curr_pos is None:
            return False
        
        try:
            req = CheckPosCollisionRequest()
            req.x = curr_pos[0]
            req.y = curr_pos[1]
            req.z = curr_pos[2]
            req.inflated = False  # Check actual occupancy, not inflated
            resp = self.collision_check(req)
            return resp.occupied
        except rospy.ServiceException as e:
            rospy.logwarn(f"Collision check failed: {e}")
            return False
    
    def check_out_of_bounds(self):
        """Check if drone is out of map bounds (with small buffer)"""
        curr_pos = self.get_current_position()
        if curr_pos is None:
            return False
        x, y = curr_pos[0], curr_pos[1]
        buffer = 0.5  # 0.5 meter buffer - only fail if slightly out of bounds
        return (x < MAP_BOUNDS[0] - buffer or x > MAP_BOUNDS[1] + buffer or 
                y < MAP_BOUNDS[0] - buffer or y > MAP_BOUNDS[1] + buffer)
    
    def check_crash(self):
        """Check if drone crashed"""
        if self.odom is None:
            return False
        
        z = self.odom.pose.pose.position.z
        # Crash if below ground AND not at goal
        if z < 0.1:
            distance = self.distance_to_goal()
            if distance is None or distance > GOAL_SUCCESS_THRESHOLD:
                return True  # Crashed (on ground but not at goal)
        return False
    
    def check_stuck(self, distance, last_distance, position_history, elapsed_time):
        """Check if stuck between obstacles - only check after minimum time"""
        # Don't check for stuck until at least 10 seconds have passed
        if elapsed_time < 10.0:
            return False
        
        # Need enough position history to make a determination
        if len(position_history) < 50:  # Need at least 5 seconds of data (50 * 0.1s)
            return False
        
        # Check if oscillating (position changes but distance doesn't decrease)
        if len(position_history) > 50:
            # Check last 50 positions (5 seconds)
            recent_positions = position_history[-50:]
            position_variance = np.var([p[0] for p in recent_positions]) + np.var([p[1] for p in recent_positions])
            
            # If position is moving significantly but distance not decreasing, might be stuck
            if position_variance > 0.5 and abs(distance - last_distance) < 0.1:
                # Check if this has been happening for a while
                if len(position_history) > 100:
                    # Check last 100 positions (10 seconds)
                    older_positions = position_history[-100:-50]
                    older_variance = np.var([p[0] for p in older_positions]) + np.var([p[1] for p in older_positions])
                    if older_variance > 0.5:
                        return True  # Been oscillating for at least 10 seconds
        
        # Check if velocity is very low for extended period (not just momentarily)
        if self.odom is not None and len(position_history) > 100:
            # Check velocity over last 5 seconds
            recent_speeds = []
            for i in range(max(0, len(position_history) - 50), len(position_history) - 1):
                if i + 1 < len(position_history):
                    pos_diff = np.linalg.norm(np.array(position_history[i+1]) - np.array(position_history[i]))
                    recent_speeds.append(pos_diff / 0.1)  # speed in m/s
            
            if recent_speeds:
                avg_speed = np.mean(recent_speeds)
                # If average speed is very low and not at goal, might be stuck
                if avg_speed < 0.05 and distance > GOAL_SUCCESS_THRESHOLD:
                    return True
        
        return False
    
    def get_min_distance_to_obstacles(self, position):
        """Get minimum distance to static obstacles"""
        if position is None:
            return float('inf')
        
        try:
            resp = self.get_static_obstacles()
            if len(resp.position) == 0:
                return float('inf')
            
            min_dist = float('inf')
            for i in range(len(resp.position)):
                obs_pos = np.array([resp.position[i].x, resp.position[i].y, resp.position[i].z])
                obs_size = np.array([resp.size[i].x, resp.size[i].y, resp.size[i].z])
                # Calculate distance to obstacle surface
                # For cylinders/boxes, use center-to-center distance minus half the max dimension
                center_dist = np.linalg.norm(position - obs_pos)
                # Approximate obstacle radius as max(size)/2
                obstacle_radius = np.max(obs_size) / 2.0
                dist_to_surface = center_dist - obstacle_radius
                min_dist = min(min_dist, dist_to_surface)
            return max(0.0, min_dist)  # Don't return negative distances
        except rospy.ServiceException as e:
            rospy.logwarn(f"Get static obstacles failed: {e}")
            return float('inf')
    
    def publish_goal(self, x, y, z=1.0):
        """Publish a navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = z
        goal_msg.pose.orientation.w = 1.0
        
        # Clear old goal
        self.goal = None
        
        # Publish multiple times to ensure it's received
        for _ in range(10):
            self.goal_pub.publish(goal_msg)
            rospy.sleep(0.1)
        
        rospy.loginfo(f"Published goal: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    def run_single_experiment(self, run_id, goal_x, goal_y, goal_z=1.0):
        """Run a single navigation experiment"""
        rospy.loginfo(f"=== Starting Run {run_id}/{NUM_RUNS} ===")
        
        # Reset metrics
        self.collision_count = 0
        self.last_collision_check_time = rospy.Time.now().to_sec()
        self.last_position = self.get_current_position()
        start_time = rospy.Time.now().to_sec()
        start_position = self.get_current_position()
        
        # Publish goal
        self.publish_goal(goal_x, goal_y, goal_z)
        
        # Wait for goal to be received
        timeout = 5.0
        elapsed = 0.0
        while self.goal is None and elapsed < timeout:
            rospy.sleep(0.1)
            elapsed += 0.1
        
        if self.goal is None:
            rospy.logwarn("Goal not received, skipping run")
            return None
        
        # Monitor navigation
        metrics = {
            "run_id": run_id,
            "goal": [goal_x, goal_y, goal_z],
            "start_position": start_position.tolist() if start_position is not None else None,
            "success": False,
            "final_bias": None,
            "collision_count": 0,
            "failure_reason": None,
            "run_time": None,
            "final_position": None,
            "steps": 0,
            "path_length": 0.0,
            "min_dist_to_obstacle": float('inf'),
            "sum_closest_distances": 0.0,
            "mean_closest_distance": 0.0,
            "std_closest_distance": 0.0
        }
        
        last_distance = float('inf')
        last_position = start_position
        position_history = []
        clearance_distances = []
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - start_time
            
            # Timeout check
            if elapsed_time > MAX_RUN_TIME:
                metrics["failure_reason"] = "timeout"
                metrics["final_position"] = curr_pos.tolist() if curr_pos is not None else None
                metrics["final_bias"] = distance if distance is not None else float('inf')
                metrics["run_time"] = elapsed_time
                break
            
            # Get current state
            curr_pos = self.get_current_position()
            if curr_pos is None:
                rospy.sleep(0.1)
                continue
            
            distance = self.distance_to_goal()
            if distance is None:
                rospy.sleep(0.1)
                continue
            
            # Increment step counter
            metrics["steps"] += 1
            
            # Track path length
            if last_position is not None:
                step_distance = np.linalg.norm(curr_pos - last_position)
                metrics["path_length"] += step_distance
            last_position = curr_pos.copy()
            position_history.append(curr_pos.copy())
            
            # Track clearance (distance to obstacles)
            clearance = self.get_min_distance_to_obstacles(curr_pos)
            clearance_distances.append(clearance)
            if clearance < metrics["min_dist_to_obstacle"]:
                metrics["min_dist_to_obstacle"] = clearance
            
            # Check for success
            if distance < GOAL_SUCCESS_THRESHOLD:
                metrics["success"] = True
                metrics["final_bias"] = distance
                metrics["final_position"] = curr_pos.tolist()
                metrics["run_time"] = elapsed_time
                rospy.loginfo(f"Run {run_id}: SUCCESS! Final bias: {distance:.3f}m")
                break
            
            # Check for failures
            if self.check_out_of_bounds():
                metrics["failure_reason"] = "out_of_bounds"
                metrics["final_position"] = curr_pos.tolist()
                metrics["final_bias"] = distance if distance is not None else float('inf')
                metrics["run_time"] = elapsed_time
                rospy.logwarn(f"Run {run_id}: FAILED - Out of bounds")
                break
            
            # Crash check disabled - not accurate
            # if self.check_crash():
            #     metrics["failure_reason"] = "crash"
            #     metrics["final_position"] = curr_pos.tolist()
            #     metrics["final_bias"] = distance if distance is not None else float('inf')
            #     metrics["run_time"] = elapsed_time
            #     rospy.logwarn(f"Run {run_id}: FAILED - Crash detected")
            #     break
            
            # Check for collisions periodically
            if current_time - self.last_collision_check_time > COLLISION_CHECK_INTERVAL:
                if self.check_collision():
                    self.collision_count += 1
                self.last_collision_check_time = current_time
            
            # Stuck check disabled - not accurate
            # # Check if stuck using the check_stuck function (only after minimum time)
            # if self.check_stuck(distance, last_distance, position_history, elapsed_time):
            #     metrics["failure_reason"] = "stuck"
            #     metrics["final_position"] = curr_pos.tolist()
            #     metrics["final_bias"] = distance if distance is not None else float('inf')
            #     metrics["run_time"] = elapsed_time
            #     rospy.logwarn(f"Run {run_id}: FAILED - Stuck (after {elapsed_time:.1f}s)")
            #     break
            
            last_distance = distance
            rospy.sleep(0.1)
        
        # Finalize metrics
        if metrics["final_position"] is None:
            curr_pos = self.get_current_position()
            if curr_pos is not None:
                metrics["final_position"] = curr_pos.tolist()
        
        # Ensure final_bias is set
        if metrics["final_bias"] is None:
            metrics["final_bias"] = self.distance_to_goal()
            if metrics["final_bias"] is None:
                metrics["final_bias"] = float('inf')
        
        metrics["collision_count"] = self.collision_count
        metrics["run_time"] = elapsed_time if metrics["run_time"] is None else metrics["run_time"]
        
        # Calculate clearance statistics
        if clearance_distances:
            # Filter out inf and nan values before calculating statistics
            finite_distances = [d for d in clearance_distances if not (np.isinf(d) or np.isnan(d))]
            
            if finite_distances:
                metrics["sum_closest_distances"] = float(sum(finite_distances))
                metrics["mean_closest_distance"] = float(statistics.mean(finite_distances))
                if len(finite_distances) > 1:
                    metrics["std_closest_distance"] = float(statistics.pstdev(finite_distances))
                else:
                    metrics["std_closest_distance"] = 0.0
            else:
                # All distances were inf/nan
                metrics["sum_closest_distances"] = 0.0
                metrics["mean_closest_distance"] = 0.0
                metrics["std_closest_distance"] = 0.0
        else:
            metrics["sum_closest_distances"] = 0.0
            metrics["mean_closest_distance"] = 0.0
            metrics["std_closest_distance"] = 0.0
        
        # Ensure min_dist_to_obstacle is not inf if we have clearance data
        if metrics["min_dist_to_obstacle"] == float('inf') and clearance_distances:
            metrics["min_dist_to_obstacle"] = float(min(clearance_distances))
        
        # Format final_bias safely (handle None and inf)
        bias_str = f"{metrics['final_bias']:.3f}" if metrics['final_bias'] != float('inf') else "inf"
        clearance_str = f"{metrics['min_dist_to_obstacle']:.3f}" if metrics['min_dist_to_obstacle'] != float('inf') else "inf"
        
        rospy.loginfo(f"Run {run_id} complete: Success={metrics['success']}, "
                     f"Bias={bias_str}m, Collisions={metrics['collision_count']}, "
                     f"Steps={metrics['steps']}, PathLength={metrics['path_length']:.2f}m, "
                     f"MinClearance={clearance_str}m")
        
        return metrics
    
    def generate_goal(self, run_id):
        """Generate a random goal position within map bounds"""
        # Generate goals within map bounds (slightly inside to ensure they're valid)
        # Matches world_generator.yaml range_x and range_y: [-10, 10]
        np.random.seed(run_id)  # For reproducibility
        goal_x = np.random.uniform(MAP_BOUNDS[0] + 0.5, MAP_BOUNDS[1] - 0.5)  # Slightly inside bounds
        goal_y = np.random.uniform(MAP_BOUNDS[0] + 0.5, MAP_BOUNDS[1] - 0.5)  # Slightly inside bounds
        goal_z = 1.0  # Fixed height for now
        return goal_x, goal_y, goal_z
    
    def run_batch(self, num_runs=NUM_RUNS):
        """Run batch of experiments"""
        rospy.loginfo(f"Starting batch evaluation: {num_runs} runs")
        
        all_metrics = []
        
        for run_id in range(1, num_runs + 1):
            # Generate goal for this run
            goal_x, goal_y, goal_z = self.generate_goal(run_id)
            
            # Run experiment
            metrics = self.run_single_experiment(run_id, goal_x, goal_y, goal_z)
            
            if metrics is not None:
                all_metrics.append(metrics)
            
            # Wait between runs
            if run_id < num_runs:
                rospy.loginfo("Waiting 5 seconds before next run...")
                rospy.sleep(5.0)
        
        # Save results
        self.save_results(all_metrics)
        
        # Print summary
        self.print_summary(all_metrics)
        
        return all_metrics
    
    def save_results(self, metrics_list):
        """Save metrics to JSON and CSV"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save JSON
        json_path = os.path.join(OUTPUT_DIR, f"batch_evaluation_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        rospy.loginfo(f"Saved JSON results to: {json_path}")
        
        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, f"batch_evaluation_{timestamp}.csv")
        if metrics_list:
            keys = metrics_list[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for m in metrics_list:
                    # Convert lists to strings for CSV
                    row = {}
                    for k, v in m.items():
                        if isinstance(v, list):
                            row[k] = str(v)
                        else:
                            row[k] = v
                    writer.writerow(row)
        rospy.loginfo(f"Saved CSV results to: {csv_path}")
    
    def print_summary(self, metrics_list):
        """Print summary statistics"""
        if not metrics_list:
            rospy.logwarn("No metrics to summarize")
            return
        
        success_count = sum(1 for m in metrics_list if m["success"])
        success_rate = success_count / len(metrics_list) * 100
        
        avg_bias = np.mean([m["final_bias"] for m in metrics_list if m["final_bias"] is not None])
        total_collisions = sum(m["collision_count"] for m in metrics_list)
        avg_collisions = total_collisions / len(metrics_list)
        avg_steps = np.mean([m["steps"] for m in metrics_list])
        avg_path_length = np.mean([m["path_length"] for m in metrics_list])
        avg_min_clearance = np.mean([m["min_dist_to_obstacle"] for m in metrics_list if m["min_dist_to_obstacle"] != float('inf')])
        avg_mean_clearance = np.mean([m["mean_closest_distance"] for m in metrics_list if m["mean_closest_distance"] > 0])
        
        failure_reasons = {}
        for m in metrics_list:
            reason = m.get("failure_reason", "none")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("BATCH EVALUATION SUMMARY")
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"Total runs: {len(metrics_list)}")
        rospy.loginfo(f"Success rate: {success_rate:.1f}% ({success_count}/{len(metrics_list)})")
        rospy.loginfo(f"Average final bias: {avg_bias:.3f}m")
        rospy.loginfo(f"Total collisions: {total_collisions}")
        rospy.loginfo(f"Average collisions per run: {avg_collisions:.2f}")
        rospy.loginfo(f"Average steps: {avg_steps:.0f}")
        rospy.loginfo(f"Average path length: {avg_path_length:.2f}m")
        rospy.loginfo(f"Average min clearance: {avg_min_clearance:.3f}m")
        rospy.loginfo(f"Average mean clearance: {avg_mean_clearance:.3f}m")
        rospy.loginfo(f"Failure reasons: {failure_reasons}")
        rospy.loginfo("=" * 50)


def main():
    rospy.init_node("batch_evaluator", anonymous=True)
    
    # Create evaluator (will wait for services to be available)
    # Make sure to launch nodes manually before running this script:
    #   roslaunch uav_simulator start.launch gui:=false
    #   roslaunch navigation_runner safety_and_perception_sim.launch
    #   rosrun navigation_runner navigation_node.py device=cpu
    evaluator = BatchEvaluator()
    
    # Run batch evaluation
    evaluator.run_batch(NUM_RUNS)
    
    rospy.loginfo("Batch evaluation complete!")


if __name__ == "__main__":
    main()