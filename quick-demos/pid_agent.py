import numpy as np
from planners.astar_planner import AStarPlanner
from planners.rrt_planner import RRTPlanner
from controllers.pid_controller import PIDController

class PIDAgent:
    """
    Wrapper agent combining a planner (e.g. A*) and a controller (e.g. PID).
    Public interface remains the same:
        pid_agent.plan(robot_state, static_obs_input, dyn_obs_input, target_dir_tensor)
    """

    def __init__(self,
                 device=None,
                 planner_type="rrt",
                 controller_type="pid",
                 max_speed=0.8,
                 grid_res=0.5,
                 half_size=20.0,
                 robot_radius=0.3,
                 kp=1.2, ki=0.0, kd=0.2):

        self.device = device
        self.max_speed = max_speed
        self.grid_res = grid_res
        self.half_size = half_size
        self.robot_radius = robot_radius

        # --- Initialize planner ---
        if planner_type == "astar":
            self.planner = AStarPlanner(grid_res=grid_res,
                                        half_size=half_size,
                                        robot_radius=robot_radius,
                                        max_speed=max_speed)
        elif planner_type == "rrt":
            self.planner = RRTPlanner(robot_radius=robot_radius, max_speed=max_speed)
        else:
            raise ValueError(f"Unsupported planner type: {planner_type}")

        # --- Initialize controller ---
        if controller_type == "pid":
            self.controller = PIDController(kp=kp, ki=ki, kd=kd, max_speed=max_speed)
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")

        # --- Track last velocity for PID derivative term ---
        self._last_vel = np.zeros(2, dtype=float)

    def reset_pid(self):
        """Reset PID internal states (for new episode)."""
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        self._last_vel[:] = 0.0

    # --- Optional global planning (A*) ---
    def compute_global_plan(self, obstacles, start, goal):
        """Compute and store global waypoints using the planner."""
        if hasattr(self.planner, "compute_global_plan"):
            self.planner.compute_global_plan(obstacles, start, goal)

    # --- Main plan interface ---
    def plan(self, robot_state, static_obs_input, dyn_obs_input, target_dir_tensor, global_goal):
        """
        1. Use planner to propose a desired velocity.
        2. Use controller to smooth the motion.
        Returns: np.array (2,) -> commanded velocity.
        """
        # Step 1: Planner proposes desired velocity
        target_dir = self.planner.local_plan(robot_state, static_obs_input,
                                     dyn_obs_input, target_dir_tensor, global_goal)

        # interpret planner output as a desired direction vector
        desired_vel = target_dir / (np.linalg.norm(target_dir) + 1e-6) * self.max_speed


        # Step 2: PID controller smooths it
        cmd_vel = self.controller.compute(desired_vel, self._last_vel, dt=0.1)

        # Update stored velocity for next iteration
        self._last_vel = cmd_vel.copy()
        return cmd_vel


# convenience alias for backward compatibility
Agent = PIDAgent
