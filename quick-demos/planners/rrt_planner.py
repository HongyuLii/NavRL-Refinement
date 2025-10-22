import numpy as np

class RRTPlanner:
    def __init__(self, step_size=0.5, max_iter=500, goal_sample_rate=0.1,
                 local_horizon=5.0, robot_radius=0.3, max_speed=0.8):
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.local_horizon = local_horizon
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.path = []

    def local_plan(self, robot_state, static_obs_input, dyn_obs_input, target_dir_tensor):
        pos = self._extract_position(robot_state)
        obstacles = self._convert_raycast_to_obstacles(pos, static_obs_input)
        goal = self._compute_local_goal(pos, target_dir_tensor, obstacles)

        self.path = self._rrt(pos, goal, obstacles)
        if not self.path:
            return self._fallback_direct(target_dir_tensor)
        return self._pure_pursuit(pos)

    # ---------------- Core RRT ----------------
    def _rrt(self, start, goal, obstacles):
        nodes = [start]
        parents = {tuple(start): None}

        def sample():
            if np.random.rand() < self.goal_sample_rate:
                return goal
            theta = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, self.local_horizon)
            return start + r * np.array([np.cos(theta), np.sin(theta)])

        for _ in range(self.max_iter):
            rand = sample()
            nearest = min(nodes, key=lambda n: np.linalg.norm(rand - n))
            direction = rand - nearest
            dist = np.linalg.norm(direction)
            if dist == 0: 
                continue
            new = nearest + self.step_size * direction / dist
            if not self._collision(nearest, new, obstacles):
                nodes.append(new)
                parents[tuple(new)] = nearest
                if np.linalg.norm(new - goal) < self.step_size:
                    return self._backtrace(parents, new)
        return []

    def _collision(self, p1, p2, obstacles):
        for ox, oy, r in obstacles:
            v = p2 - p1
            u = ((ox - p1[0]) * v[0] + (oy - p1[1]) * v[1]) / (np.dot(v, v) + 1e-9)
            u = np.clip(u, 0, 1)
            closest = p1 + u * v
            if np.linalg.norm(closest - np.array([ox, oy])) <= (r + self.robot_radius):
                return True
        return False

    def _backtrace(self, parents, last):
        path = [last]
        while parents[tuple(path[-1])] is not None:
            path.append(parents[tuple(path[-1])])
        return path[::-1]

    # ---------------- Helpers (same as A*) ----------------
    # _extract_position, _convert_raycast_to_obstacles, _compute_local_goal,
    # _fallback_direct, _pure_pursuit – identical to AStarPlanner versions.
