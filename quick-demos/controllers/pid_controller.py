import numpy as np
from simple_pid import PID


class PIDController:
    """PID-based smoothing controller for 2D velocity commands using simple_pid.

    compute(desired_vel, last_vel, dt) -> returns new velocity (np.array shape (2,)).
    """

    def __init__(self, kp=1.2, ki=0.0, kd=0.2, max_speed=0.8, acc_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_speed = float(max_speed)
        self.acc_limit = float(acc_limit)
        self.reset()

    def reset(self):
        # create fresh PID objects per axis
        self.pid_x = PID(self.kp, self.ki, self.kd, setpoint=0.0)
        self.pid_y = PID(self.kp, self.ki, self.kd, setpoint=0.0)
        # output limits correspond to allowed acceleration (m/s^2)
        self.pid_x.output_limits = (-self.acc_limit, self.acc_limit)
        self.pid_y.output_limits = (-self.acc_limit, self.acc_limit)

    def compute(self, desired_vel, last_vel, dt=0.1):
        """Compute a smoothed velocity command.

        - desired_vel: array-like (2,) desired velocity (m/s)
        - last_vel: array-like (2,) previous commanded velocity (m/s)
        - dt: timestep (s)
        """
        # Convert inputs to numpy
        des = np.asarray(desired_vel, dtype=float).reshape(-1)[:2]
        last = np.asarray(last_vel, dtype=float).reshape(-1)[:2]
        dt = float(dt) if dt > 0 else 1e-6

        # set desired setpoints and compute acceleration outputs
        self.pid_x.setpoint = float(des[0])
        self.pid_y.setpoint = float(des[1])
        ax = float(self.pid_x(last[0]))
        ay = float(self.pid_y(last[1]))

        # integrate acceleration -> velocity
        new_vx = last[0] + ax * dt
        new_vy = last[1] + ay * dt

        # clamp speed
        speed = np.hypot(new_vx, new_vy)
        if speed > self.max_speed and speed > 1e-6:
            scale = self.max_speed / speed
            new_vx *= scale
            new_vy *= scale

        return np.array([float(new_vx), float(new_vy)], dtype=float)


# Backwards compatibility: export name expected by pid_agent wrapper
Controller = PIDController
