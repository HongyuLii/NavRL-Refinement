import numpy as np
import heapq

class PIDAgent:
    """Simple agent with planner + PID controller.
    - compute_global_plan(obstacles, start, goal) builds a coarse occupancy grid and runs A*.
    - plan(robot_state, static_obs_input, dyn_obs_input, target_dir_tensor) returns a 2D velocity np.array.
    Usage:
      agent = PIDAgent(max_speed=0.8)
      agent.compute_global_plan(obstacles, start, goal)  # optional
      v = agent.plan(robot_state, static_obs_input, dyn_obs_input, target_dir_tensor)
    """

    def __init__(self, device=None, kp=1.2, ki=0.0, kd=0.2, max_speed=0.8, grid_res=0.5, half_size=20.0, robot_radius=0.3):
        self.device = device
        self.pid_Kp = kp
        self.pid_Ki = ki
        self.pid_Kd = kd
        self.max_speed = max_speed
        self.grid_res = grid_res
        self.half_size = half_size
        self.robot_radius = robot_radius

        # PID state
        self._pid_integral = np.zeros(2, dtype=float)
        self._pid_prev_err = np.zeros(2, dtype=float)
        self._pid_last_vel = np.zeros(2, dtype=float)

        # planner state
        self.waypoints = []  # list of np.array([x,y])
        self.wp_tol = 0.25

    def reset_pid(self):
        self._pid_integral[:] = 0.0
        self._pid_prev_err[:] = 0.0
        self._pid_last_vel[:] = 0.0

    # ---------------------- Planner (coarse occupancy + A*) ----------------------
    def build_occupancy_grid(self, obstacles, grid_res=None, half_size=None):
        if grid_res is None:
            grid_res = self.grid_res
        if half_size is None:
            half_size = self.half_size
        size = int((2*half_size) / grid_res)
        origin = -half_size
        grid = np.zeros((size, size), dtype=np.uint8)
        for ox, oy, r in obstacles:
            minx = int(max(0, np.floor((ox - r - self.robot_radius - origin) / grid_res)))
            maxx = int(min(size-1, np.ceil((ox + r + self.robot_radius - origin) / grid_res)))
            miny = int(max(0, np.floor((oy - r - self.robot_radius - origin) / grid_res)))
            maxy = int(min(size-1, np.ceil((oy + r + self.robot_radius - origin) / grid_res)))
            for i in range(minx, maxx+1):
                for j in range(miny, maxy+1):
                    cx = origin + (i + 0.5) * grid_res
                    cy = origin + (j + 0.5) * grid_res
                    if (cx - ox)**2 + (cy - oy)**2 <= (r + self.robot_radius)**2:
                        grid[j, i] = 1
        return grid, grid_res, origin

    def a_star(self, start_xy, goal_xy, occ_grid, grid_res, origin):
        def to_cell(p):
            return (int((p[0]-origin)/grid_res), int((p[1]-origin)/grid_res))
        start = to_cell(start_xy)
        goal = to_cell(goal_xy)
        # handle out-of-bounds
        h = lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1])
        neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        open_heap = [(0 + h(start, goal), 0, start, None)]
        came_from = {}
        cost_so_far = {start:0}
        closed = set()
        width = occ_grid.shape[1]
        height = occ_grid.shape[0]
        while open_heap:
            _, cost, current, parent = heapq.heappop(open_heap)
            if current in closed:
                continue
            came_from[current] = parent
            if current == goal:
                break
            closed.add(current)
            for dx,dy in neighbors:
                nxt = (current[0]+dx, current[1]+dy)
                if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= width or nxt[1] >= height:
                    continue
                if occ_grid[nxt[1], nxt[0]] == 1:
                    continue
                new_cost = cost_so_far[current] + np.hypot(dx,dy)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + h(nxt, goal)
                    heapq.heappush(open_heap, (priority, new_cost, nxt, current))
        if goal not in came_from:
            return []
        path = []
        cur = goal
        while cur is not None:
            cx = origin + (cur[0] + 0.5) * grid_res
            cy = origin + (cur[1] + 0.5) * grid_res
            path.append(np.array([cx, cy], dtype=float))
            cur = came_from.get(cur)
        path.reverse()
        return path

    def compute_global_plan(self, obstacles, start_xy, goal_xy, grid_res=None):
        occ_grid, grid_res, origin = self.build_occupancy_grid(obstacles, grid_res=grid_res)
        path = self.a_star(start_xy, goal_xy, occ_grid, grid_res, origin)
        self.waypoints = path
        return path

    # ---------------------- Controller / plan interface ----------------------
    def pure_pursuit_velocity(self, robot_pos, waypoints, max_speed=None, wp_tol=None):
        if max_speed is None:
            max_speed = self.max_speed
        if wp_tol is None:
            wp_tol = self.wp_tol
        if not waypoints:
            return np.zeros(2, dtype=float)
        # drop reached waypoints
        while waypoints and np.linalg.norm(waypoints[0] - robot_pos) < wp_tol:
            waypoints.pop(0)
        if not waypoints:
            return np.zeros(2, dtype=float)
        dir_vec = waypoints[0] - robot_pos
        dist = np.linalg.norm(dir_vec)
        if dist < 1e-6:
            return np.zeros(2, dtype=float)
        vel = (dir_vec / dist) * min(max_speed, dist / 0.1)
        return vel

    def pid_smooth(self, desired_vel, dt=0.1):
        err = desired_vel - self._pid_last_vel
        self._pid_integral += err * dt
        deriv = (err - self._pid_prev_err) / (dt if dt > 0 else 1e-6)
        output = (self.pid_Kp * err) + (self.pid_Ki * self._pid_integral) + (self.pid_Kd * deriv)
        cmd_vel = self._pid_last_vel + output
        speed = np.linalg.norm(cmd_vel)
        if speed > self.max_speed:
            cmd_vel = cmd_vel / speed * self.max_speed
        self._pid_prev_err = err
        self._pid_last_vel = cmd_vel.copy()
        return cmd_vel

    def plan(self, robot_state, static_obs_input, dyn_obs_input, target_dir_tensor):
        """Plan interface matching existing agent.plan signature.
        - robot_state, static_obs_input, dyn_obs_input are accepted but not required for this simple controller.
        - If self.waypoints exists, follow waypoints; otherwise follow target_dir_tensor (goal vector).
        Returns: numpy array shape (2,) velocity in m/s.
        """
        # extract robot_pos or target from robot_state / target_dir_tensor
        try:
            # target_dir_tensor shapes vary; try to extract first two entries
            if hasattr(target_dir_tensor, 'cpu') and hasattr(target_dir_tensor, 'numpy'):
                tgt = target_dir_tensor.cpu().numpy().reshape(-1)
            else:
                tgt = np.asarray(target_dir_tensor).reshape(-1)
            if tgt.size >= 2:
                target_vec = np.array(tgt[:2], dtype=float)
            else:
                target_vec = np.zeros(2, dtype=float)
        except Exception:
            target_vec = np.zeros(2, dtype=float)

        # get robot position from robot_state if possible (fallback required)
        robot_pos = None
        try:
            # robot_state may be a tensor or numpy with [x,y,...]
            if hasattr(robot_state, 'cpu') and hasattr(robot_state, 'numpy'):
                rs = robot_state.cpu().numpy().reshape(-1)
                robot_pos = np.array([rs[0], rs[1]], dtype=float)
        except Exception:
            robot_pos = None

        # If no robot_pos, assume origin (0,0) -- caller should call compute_global_plan and pass absolute waypoints
        if robot_pos is None:
            robot_pos = np.array([0.0, 0.0], dtype=float)

        # Follow waypoints if present
        if self.waypoints:
            vel_des = self.pure_pursuit_velocity(robot_pos, self.waypoints, max_speed=self.max_speed)
        else:
            # use target_vec direction
            norm = np.linalg.norm(target_vec)
            if norm < 1e-6:
                vel_des = np.zeros(2, dtype=float)
            else:
                vel_des = (target_vec / norm) * self.max_speed

        # PID smoothing
        vel_cmd = self.pid_smooth(vel_des, dt=0.1)
        return vel_cmd


# convenience alias for compatibility
Agent = PIDAgent
