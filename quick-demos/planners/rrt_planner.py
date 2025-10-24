import numpy as np

class RRTPlanner:
    """
    Local RRT planner (PPO-aligned, safe version)
    ----------------------------------------------------------
    - Goal-biased random sampling (only from free space)
    - Rejects samples/new nodes inside obstacles
    - Checks edge collisions before expanding tree
    - Handles direct goal motion when visible/unblocked
    - Matches PPO motion scale (step_size ≈ 0.2 m per 0.1 s)
    """

    def __init__(self, step_size=0.2, max_iter=1500, goal_sample_rate=0.2,
                 local_horizon=5.0, robot_radius=0.3, max_speed=2.0):
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.local_horizon = local_horizon
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.path = []

    # ------------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------------
    def local_plan(self, robot_state, static_obs_input, dyn_obs_input,
                target_dir_tensor, global_goal=None):
        pos = self._extract_position(robot_state)

        # --- Compute direction from target_dir_tensor ---
        try:
            if hasattr(target_dir_tensor, "cpu"):
                dir_vec = target_dir_tensor.cpu().numpy().reshape(-1)[:2]
            else:
                dir_vec = np.asarray(target_dir_tensor).reshape(-1)[:2]
        except Exception:
            dir_vec = np.zeros(2)
        start_angle = float(np.arctan2(dir_vec[1], dir_vec[0])) if np.linalg.norm(dir_vec) > 1e-6 else 0.0

        # --- Convert lidar-like input to obstacles ---
        obstacles = self._convert_raycast_to_obstacles(pos, static_obs_input, start_angle=start_angle)

        # =====================================================
        # ✅ Helper: internal debug function for path collisions
        # =====================================================
        def _debug_path_check(label):
            if getattr(self, "path", None):
                collided, clearance = self.debug_path_collision(self.path, obstacles, verbose=False)
                print(f"[RRT DEBUG] {label}: collided={collided}, min_clearance={clearance:.3f} m, "
                    f"path_len={len(self.path)}")

        # --- If global goal is known, handle directly ---
        if global_goal is not None:
            dist_to_goal = np.linalg.norm(global_goal - pos)

            # Case 1: very close and visible → move directly
            if dist_to_goal <= self.step_size and not self._collision(pos, global_goal, obstacles):
                dir_vec = global_goal - pos
                return (dir_vec / dist_to_goal) * min(self.max_speed, dist_to_goal / 0.1)

            # Case 2: goal within local horizon
            if dist_to_goal <= self.local_horizon:
                if not self._collision(pos, global_goal, obstacles):
                    dir_vec = global_goal - pos
                    dist = np.linalg.norm(dir_vec)
                    return (dir_vec / dist) * min(self.max_speed, dist / 0.1)
                else:
                    # Blocked — run RRT toward goal
                    self.path = self._rrt(pos, global_goal, obstacles)
                    _debug_path_check("global_goal")  # <── ADD HERE
                    if self.path:
                        cmd = self._pure_pursuit(pos, obstacles)
                        if self._collision(pos, pos + cmd * 0.5, obstacles):
                            cmd = self._fallback_direct(target_dir_tensor)
                        return cmd
                    return self._fallback_direct(target_dir_tensor)

        # --- Otherwise: plan toward a projected local goal ---
        local_goal = self._compute_local_goal(pos, target_dir_tensor, obstacles)
        self.path = self._rrt(pos, local_goal, obstacles)
        _debug_path_check("local_goal")  # <── ADD HERE

        if not self.path:
            return self._fallback_direct(target_dir_tensor)

        cmd = self._pure_pursuit(pos, obstacles)
        if self._collision(pos, pos + cmd * 0.5, obstacles):
            cmd = self._fallback_direct(target_dir_tensor)
        return cmd


    # ------------------------------------------------------------
    # CORE RRT (SAFE VERSION)
    # ------------------------------------------------------------
    def _rrt(self, start, goal, obstacles):
        nodes = [start]
        parents = {tuple(start): None}

        def sample_free():
            """Sample a random point in free space."""
            for _ in range(50):  # retry until a free point is found
                if np.random.rand() < self.goal_sample_rate:
                    pt = goal
                else:
                    theta = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(0, self.local_horizon)
                    pt = start + r * np.array([np.cos(theta), np.sin(theta)])
                if not self._point_in_obstacle(pt, obstacles):
                    return pt
            return start  # fallback if all fail

        for _ in range(self.max_iter):
            rand = sample_free()
            nearest = min(nodes, key=lambda n: np.linalg.norm(rand - n))
            direction = rand - nearest
            dist = np.linalg.norm(direction)
            if dist == 0:
                continue

            step = min(self.step_size, dist)
            new = nearest + step * direction / dist

            # --- Reject if new node or edge collides ---
            if self._point_in_obstacle(new, obstacles):
                continue
            if self._collision(nearest, new, obstacles):
                continue

            nodes.append(new)
            parents[tuple(new)] = nearest

            if np.linalg.norm(new - goal) < self.step_size and not self._collision(new, goal, obstacles):
                return self._backtrace(parents, new)
        return []

    # ------------------------------------------------------------
    # COLLISION + PATH RECONSTRUCTION
    # ------------------------------------------------------------
    def _collision(self, p1, p2, obstacles, num_samples=32):
        """Check segment collision with circular obstacles (fine-sampled)."""
        for i in range(num_samples + 1):
            interp = p1 + (i / num_samples) * (p2 - p1)
            for ox, oy, r in obstacles:
                if np.linalg.norm(interp - np.array([ox, oy])) <= (r + self.robot_radius):
                    return True
        return False

    def _backtrace(self, parents, last):
        path = [last]
        while parents[tuple(path[-1])] is not None:
            path.append(parents[tuple(path[-1])])
        return path[::-1]

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def _extract_position(self, robot_state):
        try:
            if hasattr(robot_state, "cpu"):
                rs = robot_state.cpu().numpy().reshape(-1)
            else:
                rs = np.asarray(robot_state).reshape(-1)
            return np.array([float(rs[0]), float(rs[1])], dtype=float)
        except Exception:
            return np.zeros(2, dtype=float)

    def _compute_local_goal(self, robot_pos, target_dir_tensor, obstacles=None):
        """Project a local goal in direction of target_dir, stepping back if blocked."""
        try:
            if hasattr(target_dir_tensor, "cpu"):
                dir_vec = target_dir_tensor.cpu().numpy().reshape(-1)[:2]
            else:
                dir_vec = np.asarray(target_dir_tensor).reshape(-1)[:2]
        except Exception:
            dir_vec = np.zeros(2)

        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
            return robot_pos
        dir_unit = dir_vec / norm
        tentative_goal = robot_pos + dir_unit * self.local_horizon

        if obstacles:
            for step in np.linspace(self.local_horizon, 0.5, 10):
                test_goal = robot_pos + dir_unit * step
                if not self._point_in_obstacle(test_goal, obstacles):
                    return test_goal
            return robot_pos
        return tentative_goal

    def _convert_raycast_to_obstacles(self, robot_pos, static_obs_input, start_angle=0.0):
        """Convert lidar/raycast 'proximity' tensor (max_range - distance) into local circular obstacles."""
        if static_obs_input is None:
            return []

        # 1) Get a clean (num_h, num_v) numpy array
        if hasattr(static_obs_input, "detach"):
            arr = static_obs_input.detach().cpu().numpy().squeeze()
        else:
            arr = np.asarray(static_obs_input).squeeze()

        if arr.ndim != 2:
            return []

        # 2) Recover true distances from inverted 'proximity'
        max_range_inferred = float(np.max(arr)) + 1e-9  # equals max_range from get_ray_cast
        dist2d = max_range_inferred - arr               # shape: (num_h, num_v)

        # 3) One distance per horizontal ray (min across vertical slices)
        ranges = np.min(dist2d, axis=1)                 # shape: (num_h,)
        num_angles = ranges.shape[0]
        if num_angles == 0:
            return []

        angles = start_angle + np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)

        # 4) Build obstacles at each hit; ignore no-hit (>= local_horizon)
        obs = []
        assumed_r = max(0.5, self.robot_radius)  # match env's min radius; OK to inflate a bit
        for i, r in enumerate(ranges):
            if not np.isfinite(r) or r <= 0.0 or r >= self.local_horizon:
                continue
            ox = robot_pos[0] + r * np.cos(angles[i])
            oy = robot_pos[1] + r * np.sin(angles[i])
            obs.append((float(ox), float(oy), float(assumed_r)))
        return obs



    def _point_in_obstacle(self, point, obstacles):
        for ox, oy, r in obstacles:
            if np.linalg.norm(point - np.array([ox, oy])) <= (r + self.robot_radius):
                return True
        return False

    # ------------------------------------------------------------
    # FALLBACKS + PURSUIT
    # ------------------------------------------------------------
    def _fallback_direct(self, target_dir_tensor):
        """Fallback straight velocity when no valid path found."""
        if hasattr(target_dir_tensor, "cpu"):
            tgt = target_dir_tensor.cpu().numpy().reshape(-1)[:2]
        else:
            tgt = np.asarray(target_dir_tensor).reshape(-1)[:2]
        norm = np.linalg.norm(tgt)
        if norm < 1e-6:
            return np.zeros(2, dtype=float)
        return (tgt / norm) * self.max_speed

    def _pure_pursuit(self, robot_pos, obstacles):
        """Follow the first visible waypoint on the path."""
        if not self.path:
            return np.zeros(2, dtype=float)

        while self.path and np.linalg.norm(self.path[0] - robot_pos) < 0.25:
            self.path.pop(0)
        if not self.path:
            return np.zeros(2, dtype=float)

        target = None
        for wp in self.path:
            if not self._collision(robot_pos, wp, obstacles):
                target = wp
                break
        if target is None:
            target = self.path[0]

        dir_vec = target - robot_pos
        dist = np.linalg.norm(dir_vec)
        if dist < 1e-6:
            return np.zeros(2, dtype=float)
        return (dir_vec / dist) * min(self.max_speed, dist / 0.1)

    def debug_path_collision(self, path, obstacles, verbose=True):
        """
        Check if the current RRT path intersects any obstacles.
        Returns (collided: bool, min_clearance: float).
        """
        if not path or len(path) < 2:
            if verbose:
                print("⚠️ No path to check.")
            return False, float('inf')

        collided = False
        min_clearance = float('inf')

        for i in range(len(path) - 1):
            p1, p2 = np.array(path[i]), np.array(path[i + 1])
            samples = np.linspace(0, 1, 200)
            for s in samples:
                pt = p1 + s * (p2 - p1)
                for ox, oy, r in obstacles:
                    dist = np.linalg.norm(pt - np.array([ox, oy])) - (r + self.robot_radius)
                    if dist < 0:
                        collided = True
                        if verbose:
                            print(f"❌ Collision on segment {i}: pt={pt}, obs=({ox:.2f},{oy:.2f},r={r:.2f})")
                    if dist < min_clearance:
                        min_clearance = dist

        if verbose:
            print(f"✅ Path check done. Collided={collided}, min_clearance={min_clearance:.3f} m")
        return collided, min_clearance

