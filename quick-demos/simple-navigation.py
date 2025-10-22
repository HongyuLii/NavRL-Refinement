import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from utils import get_robot_state, get_ray_cast
import torch
import random
from env import generate_obstacles_grid, sample_free_start, sample_free_goal
from pid_agent import PIDAgent 
import json
import csv
import time
import os
import statistics
import argparse

# === Set random seed ===
SEED = 0 
random.seed(SEED)
np.random.seed(SEED)

# === Device ===
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# === Constants ===
MAP_HALF_SIZE = 20
OBSTACLE_REGION_MIN = -15
OBSTACLE_REGION_MAX = 15
MIN_RADIUS = 0.5
MAX_RADIUS = 1.0
MAX_RAY_LENGTH = 4.0
DT = 0.1
GOAL_REACHED_THRESHOLD = 0.3
HRES_DEG = 10.0
VFOV_ANGLES_DEG = [-10.0, 0.0, 10.0, 20.0]
GRID_DIV = 7

# Add robot radius and output settings
ROBOT_RADIUS = 0.3
MAX_FRAMES = 300
OUTPUT_DIR = "run_metrics"
# separate folder for NavRL agent metrics to avoid overwriting other runs
NavRl_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "NavRL_agent")
os.makedirs(NavRl_OUTPUT_DIR, exist_ok=True)
# separate folder for PID agent metrics to avoid overwriting other runs
PID_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pid_agent")
os.makedirs(PID_OUTPUT_DIR, exist_ok=True)
# separate folder for MPC agent metrics to avoid overwriting other runs
MPC_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mpc_agent")
os.makedirs(MPC_OUTPUT_DIR, exist_ok=True)


# === Setup ===
obstacles = generate_obstacles_grid(GRID_DIV, OBSTACLE_REGION_MIN, OBSTACLE_REGION_MAX, MIN_RADIUS, MAX_RADIUS)
robot_vel = np.array([0.0, 0.0])
goal = np.array([5.0, 18.0])
robot_pos = np.array([0.0, -18.0])
start_pos = robot_pos.copy()
target_dir = goal - robot_pos 
trajectory = []

# Metrics bookkeeping
episode_metrics = {
    "start": start_pos.tolist(),
    "goal": goal.tolist(),
    "steps": 0,
    "time_s": 0.0,
    "path_length": 0.0,
    "collisions": 0,
    "min_dist_to_obstacle": float("inf"),
    "reached_goal": False,
    "timeout": False,
    "timestamp": time.strftime("%Y%m%d-%H%M%S")
}
_prev_pos = None
_episode_saved = False

# === NavRL Agent ===
pid_agent = PIDAgent(device=device, robot_radius=ROBOT_RADIUS, grid_res=0.5)


# === Visualization setup ===
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#fefcfb')  # Light warm figure background
ax.set_facecolor('#fdf6e3')         # Slightly warm off-white axes background
ax.set_xlim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_ylim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_aspect('equal')
ax.set_title("NavRL Goal Navigation")
# ax.add_patch(Rectangle(
#                     (-MAP_HALF_SIZE, OBSTACLE_REGION_MIN),
#                     2 * MAP_HALF_SIZE,
#                     OBSTACLE_REGION_MAX - OBSTACLE_REGION_MIN,
#                     edgecolor='black',
#                     facecolor='lightgray',
#                     alpha=0.3,          # Slightly stronger for contrast
#                     linewidth=1.0,      # Thinner edge
#                     linestyle='--'      # Dotted border looks more natural
#                 ))
robot_dot, = ax.plot([], [], 'o', markersize=6, color="royalblue" , label='Robot', zorder=5)
goal_dot, = ax.plot([], [], marker='*', markersize=15, color='red', linestyle='None', label='Goal')
start_dot, = ax.plot([], [], marker='s', markersize=8, color='navy', label='Start', linestyle='None', zorder=3)
trajectory_line, = ax.plot([], [], '-', linewidth=1.5, color="lime", label='Trajectory')
ray_lines = [ax.plot([], [], 'r--', linewidth=0.5)[0] for _ in range(int(360 / HRES_DEG))]
ax.legend(loc='upper left')

for obs in obstacles:
    ax.add_patch(Circle((obs[0], obs[1]), obs[2], color='gray'))

# helper: min distance to obstacles and collision check
def min_distance_to_obstacles(pos, obstacles):
    md = float("inf")
    for ox, oy, r in obstacles:
        d = np.linalg.norm(pos - np.array([ox, oy])) - r
        if d < md:
            md = d
    return md

def check_collision(pos, obstacles, robot_radius=ROBOT_RADIUS):
    for ox, oy, r in obstacles:
        if np.linalg.norm(pos - np.array([ox, oy])) <= (r + robot_radius):
            return True
    return False

def _save_metrics(metrics):
    """Save metrics dict as JSON and CSV (single-row)."""
    global _episode_saved
    if _episode_saved:
        return
    fname_base = f"metrics_{metrics['timestamp']}"
    json_path = os.path.join(PID_OUTPUT_DIR, fname_base + ".json")
    csv_path = os.path.join(PID_OUTPUT_DIR, fname_base + ".csv")
    # JSON
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    # CSV (single-row)
    keys = list(metrics.keys())
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Saved metrics -> {json_path}, {csv_path}")
    _episode_saved = True


# -----------------------
# Batch-run and clearance time-series utilities
# -----------------------

def run_episode(start_pos_in, goal_in, save_prefix=None):
    """Run one episode (headless) and return metrics dict + per-step clearance list."""
    # reset state
    robot_pos = start_pos_in.copy()
    robot_vel = np.array([0.0, 0.0])
    target_dir = goal_in - robot_pos
    trajectory = []
    _prev_pos = None
    per_step_clearance = []

    # episode metrics local copy
    metrics = {
        "start": start_pos_in.tolist(),
        "goal": goal_in.tolist(),
        "steps": 0,
        "time_s": 0.0,
        "path_length": 0.0,
        "collisions": 0,
        "min_dist_to_obstacle": float("inf"),
        "reached_goal": False,
        "timeout": False,
        "timestamp": time.strftime("%Y%m%d-%H%M%S")
    }
    # reset agent PID internal state if present
    if hasattr(pid_agent, "reset_pid"):
        try:
            pid_agent.reset_pid()
        except Exception:
            pass

    for t in range(MAX_FRAMES):
        # goal check
        to_goal = goal_in - robot_pos
        dist = np.linalg.norm(to_goal)
        if dist < GOAL_REACHED_THRESHOLD:
            metrics["reached_goal"] = True
            break

        # step bookkeeping
        metrics["steps"] += 1

        # clearance and collision
        md = min_distance_to_obstacles(robot_pos, obstacles)
        per_step_clearance.append(float(md))
        if md < metrics["min_dist_to_obstacle"]:
            metrics["min_dist_to_obstacle"] = float(md)
        if check_collision(robot_pos, obstacles):
            metrics["collisions"] += 1

        # incremental path length
        if _prev_pos is None:
            _prev_pos = robot_pos.copy()
        step_len = np.linalg.norm(robot_pos - _prev_pos)
        metrics["path_length"] += float(step_len)
        _prev_pos = robot_pos.copy()

        # create inputs and call agent
        robot_state = get_robot_state(robot_pos, goal_in, robot_vel, target_dir, device=device)
        static_obs_input, _, _ = get_ray_cast(robot_pos, obstacles, max_range=MAX_RAY_LENGTH,
                                              hres_deg=HRES_DEG, vfov_angles_deg=VFOV_ANGLES_DEG,
                                              start_angle_deg=np.degrees(np.arctan2(target_dir[1], target_dir[0])),
                                              device=device)
        dyn_obs_input = torch.zeros((1, 1, 5, 10), dtype=torch.float, device=device)
        target_dir_tensor = torch.tensor(np.append(target_dir[:2], 0.0), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

        velocity = pid_agent.plan(robot_state, static_obs_input, dyn_obs_input, target_dir_tensor)
        velocity = np.asarray(velocity, dtype=float).reshape(2,)
        print(velocity)
        # update state
        robot_pos = robot_pos + velocity * DT
        robot_vel = velocity.copy()
        trajectory.append(robot_pos.copy())

    # finalize metrics
    metrics["time_s"] = metrics["steps"] * DT
    if not metrics["reached_goal"]:
        metrics["timeout"] = True
    # additional clearance aggregates
    if per_step_clearance:
        metrics["sum_closest_distances"] = float(sum(per_step_clearance))
        metrics["mean_closest_distance"] = float(statistics.mean(per_step_clearance))
        metrics["std_closest_distance"] = float(statistics.pstdev(per_step_clearance))
    else:
        metrics["sum_closest_distances"] = 0.0
        metrics["mean_closest_distance"] = 0.0
        metrics["std_closest_distance"] = 0.0

    # save per-episode metrics + optionally the time-series clearance
    if save_prefix:
        fname_base = f"{save_prefix}_{metrics['timestamp']}"
        json_path = os.path.join(PID_OUTPUT_DIR, fname_base + ".json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        # save per-step clearance as JSON array
        ts_path = os.path.join(PID_OUTPUT_DIR, fname_base + "_clearance.json")
        with open(ts_path, "w") as f:
            json.dump({"clearance": per_step_clearance}, f)
    return metrics, per_step_clearance


def batch_run(num_episodes=30, min_clearance=0.5):
    """Run multiple episodes with different start/goal pairs and save batch summary.
    Also save a combined JSON containing per-episode metrics and clearance time-series,
    and a single CSV with one row per episode (clearance time-series stored as JSON string).
    """
    batch_results = []
    combined_entries = []
    for i in range(num_episodes):
        # sample start and goal (ensure they are not too close)
        s = sample_free_start(obstacles, goal, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
        g = sample_free_goal(obstacles, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
        # ensure reasonable separation
        if np.linalg.norm(s - g) < 5.0:
            # pick another goal until separation OK (few attempts)
            for _ in range(20):
                g = sample_free_goal(obstacles, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
                if np.linalg.norm(s - g) >= 5.0:
                    break
        # prefix = f"episode_{i+1:02d}"
        # metrics, clearance_ts = run_episode(s, g, save_prefix=prefix)
        # do not save individual per-episode files; collect metrics only
        metrics, clearance_ts = run_episode(s, g, save_prefix=None)
        batch_results.append(metrics)
        # combine metrics + clearance timeseries for a single-file export
        entry = dict(metrics)
        entry["clearance_ts"] = clearance_ts
        combined_entries.append(entry)
        print(f"Episode {i+1}/{num_episodes} -> reached: {metrics['reached_goal']}, steps: {metrics['steps']}, collisions: {metrics['collisions']}")

    # save batch summary JSON (metrics only)
    batch_ts = time.strftime("%Y%m%d-%H%M%S")
    batch_json = os.path.join(PID_OUTPUT_DIR, f"batch_summary_{batch_ts}.json")
    with open(batch_json, "w") as f:
        json.dump(batch_results, f, indent=2)

    # save combined JSON (metrics + clearance time series)
    combined_json = os.path.join(PID_OUTPUT_DIR, f"combined_episodes_{batch_ts}.json")
    with open(combined_json, "w") as f:
        json.dump(combined_entries, f, indent=2)

    # save CSV summary (one row per episode). Clearance stored as JSON string in clearance_ts column.
    batch_csv = os.path.join(PID_OUTPUT_DIR, f"batch_summary_{batch_ts}.csv")
    keys = ["start", "goal", "steps", "time_s", "path_length", "collisions", "min_dist_to_obstacle",
            "sum_closest_distances", "mean_closest_distance", "std_closest_distance", "reached_goal", "timeout", "timestamp", "clearance_ts"]
    with open(batch_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r, combined in zip(batch_results, combined_entries):
            row = {k: r.get(k, "") for k in keys}
            row["start"] = str(r["start"])
            row["goal"] = str(r["goal"])
            # put clearance_ts as compact JSON string
            row["clearance_ts"] = json.dumps(combined.get("clearance_ts", []))
            writer.writerow(row)

    print(f"Saved batch summary -> {batch_json}, {batch_csv}")
    print(f"Saved combined episodes JSON -> {combined_json}")
    return batch_results

# If you want to run a batch immediately, uncomment:
# batch_run(num_episodes=30)

# === Simulation update ===
def update(frame):
    global robot_pos, robot_vel, goal, trajectory, target_dir, start_pos, _prev_pos, episode_metrics

    # Goal reach check
    to_goal = goal - robot_pos
    dist = np.linalg.norm(to_goal)
    if dist < GOAL_REACHED_THRESHOLD:
        episode_metrics["reached_goal"] = True
        episode_metrics["time_s"] = episode_metrics["steps"] * DT
        _save_metrics(episode_metrics)
        return

    # Timeout / final frame check
    if frame >= (MAX_FRAMES - 1):
        episode_metrics["timeout"] = True
        episode_metrics["time_s"] = episode_metrics["steps"] * DT
        _save_metrics(episode_metrics)
        return

    # Update step count
    episode_metrics["steps"] += 1

    # Update min distance metric
    md = min_distance_to_obstacles(robot_pos, obstacles)
    if md < episode_metrics["min_dist_to_obstacle"]:
        episode_metrics["min_dist_to_obstacle"] = float(md)

    # Collision check
    if check_collision(robot_pos, obstacles):
        episode_metrics["collisions"] += 1

    # Compute incremental path length
    if _prev_pos is None:
        _prev_pos = robot_pos.copy()
    step_len = np.linalg.norm(robot_pos - _prev_pos)
    episode_metrics["path_length"] += float(step_len)
    _prev_pos = robot_pos.copy()


    # Get robot internal states
    robot_state = get_robot_state(robot_pos, goal, robot_vel, target_dir, device=device)

    # Get static obstacle representations
    static_obs_input, range_matrix, ray_segments = get_ray_cast(robot_pos, obstacles, max_range=MAX_RAY_LENGTH,
                                                       hres_deg=HRES_DEG,
                                                       vfov_angles_deg=VFOV_ANGLES_DEG,
                                                       start_angle_deg=np.degrees(np.arctan2(target_dir[1], target_dir[0])),
                                                       device=device)
    # Get dynamic obstacle representations (assume zero)
    dyn_obs_input = torch.zeros((1, 1, 5, 10), dtype=torch.float, device=device)

    # Target direction in tensor
    target_dir_tensor = torch.tensor(np.append(target_dir[:2], 0.0), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    # Output the planned velocity
    velocity = pid_agent.plan(robot_state, static_obs_input, dyn_obs_input, target_dir_tensor)

    # ---Visualizaton update---
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    start_dot.set_data([start_pos[0]], [start_pos[1]])
    goal_dot.set_data([goal[0]], [goal[1]])
    trajectory.append(robot_pos.copy())
    trajectory_np = np.array(trajectory)
    trajectory_line.set_data(trajectory_np[:, 0], trajectory_np[:, 1])

    # Update simulation states
    robot_pos += velocity * DT
    robot_vel = velocity.copy()

    return [robot_dot, goal_dot, trajectory_line, start_dot] + ray_lines

# replace the unconditional animation with CLI handling
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple NavRL demo: run interactive demo or batch episodes.")
    parser.add_argument("--batch", action="store_true", help="Run batch episodes headless and save metrics")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes when running batch")
    parser.add_argument("--no-gui", action="store_true", help="Run a single episode headless (no GUI) and save metrics")
    args = parser.parse_args()

    if args.batch:
        batch_run(num_episodes=args.episodes)
    else:
        if args.no_gui:
            metrics, clearance = run_episode(start_pos, goal, save_prefix="single")
            print(json.dumps(metrics, indent=2))
        else:
            ani = animation.FuncAnimation(fig, update, frames=MAX_FRAMES, interval=20, blit=False)
            plt.show()