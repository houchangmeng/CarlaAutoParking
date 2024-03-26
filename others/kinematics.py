from functools import partial
from jax import jit, jacobian
import jax.numpy as jnp
import numpy as np
from config import VehicleConfig
from others.se2state import SE2

# def kinematics(state: np.ndarray, action: np.ndarray, vehicle_cfg=VehicleConfig()):
#     x, y, heading, v = state
#     accel, delta = action

#     beta = np.arctan(vehicle_cfg.lf * delta / vehicle_cfg.wheel_base)
#     xdot = v * np.cos(heading + beta)
#     ydot = v * np.sin(heading + beta)

#     headingdot = v * np.cos(beta) / vehicle_cfg.wheel_base * np.tan(delta)

#     return np.array([xdot, ydot, headingdot, accel])


def kinematics(state: np.ndarray, action: np.ndarray, vehicle_cfg=VehicleConfig()):
    x, y, heading, v = state
    accel, delta = action

    xdot = v * np.cos(heading)
    ydot = v * np.sin(heading)

    headingdot = v / vehicle_cfg.wheel_base * np.tan(delta)

    return np.array([xdot, ydot, headingdot, accel])


def euler_step(
    state: np.ndarray, state_dot: np.ndarray, h: float, vehicle_cfg=VehicleConfig()
):
    x, y, heading, v = state
    current_se2 = SE2(x, y, heading)

    xdot, ydot, headingdot, accel = state_dot

    next_x = current_se2.x + h * xdot
    next_y = current_se2.y + h * ydot
    next_heading = current_se2.so2.heading + h * headingdot

    next_se2 = SE2(next_x, next_y, next_heading)

    x, y, heading = next_se2.x, next_se2.y, next_se2.so2.heading

    v = v + h * accel

    v = np.clip(v, -vehicle_cfg.max_v, vehicle_cfg.max_v)

    return next_x, next_y, next_heading, v


def euler_integration(
    state: np.ndarray, action: np.ndarray, h: float, vehicle_cfg=VehicleConfig()
):
    state_dot = kinematics(state, action, vehicle_cfg)
    return euler_step(state, state_dot, h, vehicle_cfg)


def runge_kutta_integration(
    state: np.ndarray, action: np.ndarray, h: float, vehicle_cfg=VehicleConfig()
):
    state = np.array(state, dtype=np.float32)

    f1 = kinematics(state, action)

    new_state = euler_step(state, f1, 0.5 * h)
    f2 = kinematics(new_state, action)

    new_state = euler_step(state, f2, 0.5 * h)
    f3 = kinematics(new_state, action)

    new_state = euler_step(state, f3, h)
    f4 = kinematics(new_state, action)

    state_dot = f1 + 2 * f2 + 2 * f3 + f4
    state = euler_step(state, state_dot, 1.0 / 6 * h)

    x, y, heading, v = state

    v = np.clip(v, -vehicle_cfg.max_v, vehicle_cfg.max_v)

    return x, y, heading, v


class CarKinematics:
    """
    ### Description

    This class is created for auto jacbian.
    """

    def __init__(self) -> None:
        self.lf = 1.5
        self.lr = 1.0
        self.wheelbase = self.lr + self.lf  # Model3 2875
        self.dt = 0.02

        self.dfdx = jacobian(self.rk4, 0)
        self.dfdu = jacobian(self.rk4, 1)
        self.statelb = jnp.array([0, 0, 0, -5])
        self.stateub = jnp.array([18, 14, 2 * jnp.pi, -5])

    @partial(jit, static_argnums=(0,))
    def kinematics(self, state, action):
        """
        TODO: need add limit at state dot.
        """
        x, y, theta, v = state
        accel, steer = action

        beta = jnp.arctan(self.lf * steer / self.wheelbase)
        xdot = v * jnp.cos(theta + beta)
        ydot = v * jnp.sin(theta + beta)
        thetadot = v * jnp.cos(beta) / self.wheelbase * jnp.tan(steer)

        return jnp.array([xdot, ydot, thetadot, accel])

    @partial(jit, static_argnums=(0,))
    def rk4(self, state, action):
        f1 = self.kinematics(state, action)
        f2 = self.kinematics(state + 0.5 * self.dt * f1, action)
        f3 = self.kinematics(state + 0.5 * self.dt * f2, action)
        f4 = self.kinematics(state + self.dt * f3, action)

        state = state + (self.dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
        state = state.at[2].set(jnp.mod(state[2], 2 * jnp.pi))
        state = jnp.clip(state, self.statelb, self.stateub)
        return state

    @partial(jit, static_argnums=(0,))
    def get_AB_matrix(self, x, u):
        A = self.dfdx(x, u)
        B = self.dfdu(x, u)

        return A, B
