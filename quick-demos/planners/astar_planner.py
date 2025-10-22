import numpy as np
import heapq

class AStarPlanner:
    """
    A local A* planner that builds a temporary occupancy grid from local perception
    and computes a short-horizon path toward the global goal direction.
    """

    def __init__(self, grid_res=0.5, half_size=20.0, robot_radius=0.3,
                 max_speed=0.8, wp_tol=0.25, local_horizon=5.0):
        self.grid_res = grid_res
        self.half_size = half_size
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.wp_tol = wp_tol
        self.local_horizon = local_horizon
        self.waypoints = []

    # ----------------- Local planning interface -----------------
    def local_plan(self, robot_state, static_obs_input, dyn_obs_input, target_dir_tensor):
        """
        1) Build a small local occupancy grid from perception (raycast → obstacles)
        2) Compute a collision-aware local goal (step-back if inside obstacle)
        3) Run A* on this local grid
        4) Return desired velocity toward the next waypoint (pure pursuit)
        """
        # 0) Extract current position
        robot_pos = self._extract_position(robot_state)

        # Compute start angle from target_dir_tensor so ray-to-world mapping matches perception
        try:
            if hasattr(target_dir_tensor, "cpu"):
                dir_vec = target_dir_tensor.cpu().numpy().reshape(-1)[:2]
            else:
                dir_vec = np.asarray(target_dir_tensor).reshape(-1)[:2]
        except Exception:
            dir_vec = np.zeros(2)
        if np.linalg.norm(dir_vec) < 1e-6:
            start_angle = 0.0
        else:
            start_angle = float(np.arctan2(dir_vec[1], dir_vec[0]))

        # 1) Perception → approximate local obstacles (aligned with start_angle)
        obstacles = self._convert_raycast_to_obstacles(robot_pos, static_obs_input, start_angle=start_angle)

        # 2) Local goal projection with step-back if blocked (Strategy A)
        local_goal = self._compute_local_goal(robot_pos, target_dir_tensor, obstacles)

        # 3) Build local grid and run A*
        occ_grid, res, origin = self._build_grid(obstacles, center=robot_pos)
        path = self._a_star(robot_pos, local_goal, occ_grid, res, origin)

        # 4) Path → velocity (fallback to goal direction if no path)
        if not path:
            return self._fallback_direct(target_dir_tensor)
        self.waypoints = path
        
        return self._pure_pursuit(robot_pos)


    # ----------------- Helper Functions -----------------
    def _extract_position(self, robot_state):
        """Extract (x,y) position from the robot_state tensor."""
        try:
            if hasattr(robot_state, "cpu"):
                rs = robot_state.cpu().numpy().reshape(-1)
            else:
                rs = np.asarray(robot_state).reshape(-1)
            return np.array([float(rs[0]), float(rs[1])], dtype=float)
        except Exception:
            return np.zeros(2, dtype=float)

    def _compute_local_goal(self, robot_pos, target_dir_tensor, obstacles=None):
        """Project a local goal in the goal direction; back off if inside obstacle."""
        try:
            if hasattr(target_dir_tensor, "cpu"):
                dir_vec = target_dir_tensor.cpu().numpy().reshape(-1)[:2]
            else:
                dir_vec = np.asarray(target_dir_tensor).reshape(-1)[:2]
        except Exception:
            dir_vec = np.zeros(2)

        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
            return robot_pos  # no goal direction
        dir_unit = dir_vec / norm
        tentative_goal = robot_pos + dir_unit * self.local_horizon

        # Step-back strategy (Strategy A)
        if obstacles:
            for step in np.linspace(self.local_horizon, 0.5, 10):
                test_goal = robot_pos + dir_unit * step
                if not self._point_in_obstacle(test_goal, obstacles):
                    return test_goal
            # if fully blocked, just stay in place
            return robot_pos

        return tentative_goal


    def _convert_raycast_to_obstacles(self, robot_pos, static_obs_input, start_angle=0.0):
        """
        Convert raycast distances (static_obs_input) into approximate local obstacles.
        static_obs_input: expected shape (1,1,num_angles,num_elevations) or similar
        start_angle: world-frame angle corresponding to the first azimuth sample (radians)
        """
        if static_obs_input is None:
            return []
        if hasattr(static_obs_input, "cpu"):
            arr = static_obs_input.cpu().numpy().squeeze()
        else:
            arr = np.asarray(static_obs_input).squeeze()

        # Collapse any elevation axis by taking the minimum valid range per azimuth
        if arr.ndim == 1:
            ranges = arr.astype(float)
        else:
            # reduce all axes except the last one (assumed azimuth) via nanmin
            axes = tuple(range(arr.ndim - 1))
            ranges = np.nanmin(arr, axis=axes).astype(float)

        num_angles = int(len(ranges))
        if num_angles == 0:
            return []

        # map azimuth samples starting from start_angle around the circle
        angles = start_angle + np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)

        obs = []
        for i in range(num_angles):
            r = float(ranges[i])
            # skip invalid or too-far hits
            if not np.isfinite(r) or r <= 0:
                continue
            if r >= self.local_horizon:
                continue
            ox = robot_pos[0] + r * np.cos(angles[i])
            oy = robot_pos[1] + r * np.sin(angles[i])
            # inflate pseudo-obstacle by robot radius to be conservative
            obs.append((float(ox), float(oy), max(0.2, float(self.robot_radius))))
        return obs

    def _point_in_obstacle(self, point, obstacles):
        """Check if a 2D point lies inside any obstacle."""
        for ox, oy, r in obstacles:
            if np.linalg.norm(point - np.array([ox, oy])) <= (r + self.robot_radius):
                return True
        return False

    # ----------------- A* and Occupancy Grid -----------------
    def _build_grid(self, obstacles, center):
        """Build a small occupancy grid centered around the robot."""
        size = max(3, int(np.ceil((2.0 * self.local_horizon) / self.grid_res)))
        origin_x = float(center[0] - self.local_horizon)
        origin_y = float(center[1] - self.local_horizon)
        grid = np.zeros((size, size), dtype=np.uint8)

        for ox, oy, r in obstacles:
            # inflate marking by robot_radius to be conservative
            inflated = r + self.robot_radius
            minx = int(max(0, np.floor((ox - inflated - origin_x) / self.grid_res)))
            maxx = int(min(size - 1, np.ceil((ox + inflated - origin_x) / self.grid_res)))
            miny = int(max(0, np.floor((oy - inflated - origin_y) / self.grid_res)))
            maxy = int(min(size - 1, np.ceil((oy + inflated - origin_y) / self.grid_res)))
            for i in range(minx, maxx + 1):
                for j in range(miny, maxy + 1):
                    if 0 <= i < size and 0 <= j < size:
                        grid[j, i] = 1
        return grid, self.grid_res, (origin_x, origin_y)

    def _a_star(self, start_xy, goal_xy, grid, res, origin):
        def to_cell(p):
            return (int(np.floor((p[0]-origin[0])/res)), int(np.floor((p[1]-origin[1])/res)))

        start = to_cell(start_xy)
        goal = to_cell(goal_xy)
        W, H = grid.shape[1], grid.shape[0]

        # if start/goal outside grid bounds, bail out
        if not (0 <= start[0] < W and 0 <= start[1] < H):
            return []
        if not (0 <= goal[0] < W and 0 <= goal[1] < H):
            return []

        h = lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1])
        open_heap = [(h(start, goal), 0, start, None)]
        came_from, cost, closed = {}, {start: 0}, set()

        while open_heap:
            _, g, cur, parent = heapq.heappop(open_heap)
            if cur in closed:
                continue
            came_from[cur] = parent
            if cur == goal:
                break
            closed.add(cur)
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nxt = (cur[0]+dx, cur[1]+dy)
                if not (0 <= nxt[0] < W and 0 <= nxt[1] < H):
                    continue
                if grid[nxt[1], nxt[0]] == 1:
                    continue
                new_cost = cost[cur] + np.hypot(dx, dy)
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    heapq.heappush(open_heap, (new_cost + h(nxt, goal), new_cost, nxt, cur))

        if goal not in came_from:
            return []

        path, cur = [], goal
        while cur is not None:
            cx = origin[0] + (cur[0] + 0.5) * res
            cy = origin[1] + (cur[1] + 0.5) * res
            path.append(np.array([float(cx), float(cy)], dtype=float))
            cur = came_from.get(cur)
        path.reverse()
        return path

    # ----------------- Fallback + waypoint tracking -----------------
    def _fallback_direct(self, target_dir_tensor):
        if hasattr(target_dir_tensor, "cpu"):
            tgt = target_dir_tensor.cpu().numpy().reshape(-1)[:2]
        else:
            tgt = np.asarray(target_dir_tensor).reshape(-1)[:2]
        norm = np.linalg.norm(tgt)
        if norm < 1e-6:
            return np.zeros(2, dtype=float)
        return (tgt / norm) * self.max_speed

    def _pure_pursuit(self, robot_pos):
        while self.waypoints and np.linalg.norm(self.waypoints[0] - robot_pos) < self.wp_tol:
            self.waypoints.pop(0)
        if not self.waypoints:
            return np.zeros(2, dtype=float)
        dir_vec = self.waypoints[0] - robot_pos
        dist = np.linalg.norm(dir_vec)
        if dist < 1e-6:
            return np.zeros(2, dtype=float)
        return (dir_vec / dist) * min(self.max_speed, dist / 0.1)
