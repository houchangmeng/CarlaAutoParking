from typing import List
from copy import deepcopy
from CarlaAutoParking.config import ControlConfig, VehicleConfig
from CarlaAutoParking.controller.controller_utils import (
    dlqr,
    get_AB_matrix,
    get_ndarray_state_from_se2state,
    lookup,
    get_minimum_distance_states,
    get_closest_time_states,
    get_closest_time_trajectory,
)
import cvxpy as cp
import numpy as np
from CarlaAutoParking.others.se2state import SE2State

from config import VehicleConfig, ControlConfig

"""
Dynamic Lateral Controller.
"""


class DynamicsLatLQRController:
    def __init__(self, veh_cfg=VehicleConfig(), ctrl_cfg=ControlConfig()) -> None:
        self.dt = ctrl_cfg.dt
        self.veh_cfg = veh_cfg
        self.ctrl_cfg = ctrl_cfg
        self.Q = ctrl_cfg.Q

        self.R = ctrl_cfg.R

        self.last_delta = 0
        self.T = 0.8  # 1 / T filter.
        self.K_save = []
        self.v_save = []

        cf = self.veh_cfg.cf_alpha
        cr = self.veh_cfg.cr_alpha
        mass = self.veh_cfg.mass
        iz = self.veh_cfg.Iz
        lf = self.veh_cfg.lf
        lr = self.veh_cfg.lr

        self.initA = np.zeros((4, 4))
        self.initA[0, 1] = 1.0
        self.initA[1, 1] = (cf + cr) / mass
        self.initA[1, 2] = -(cf + cr) / mass
        self.initA[1, 3] = (lf * cf - lr * cr) / mass
        self.initA[2, 3] = 1.0
        self.initA[3, 1] = (lf * cf - lr * cr) / iz
        self.initA[3, 2] = -(lf * cf - lr * cr) / iz
        self.initA[3, 3] = (lf * lf * cf + lr * lr * cr) / iz

        self.initB = np.zeros((4, 1))
        self.initB[1, 0] = -cf / mass
        self.initB[3, 0] = -lf * cf / iz

        self.Kv = lr * mass / (cf * (lf + lr)) - lf * mass / (cr * (lf + lr))

    def get_AB_matrix(self, current_v):
        A = deepcopy(self.initA)

        A[1, 1] = self.initA[1, 1] / current_v
        A[1, 3] = self.initA[1, 3] / current_v

        A[3, 1] = self.initA[3, 1] / current_v
        A[3, 3] = self.initA[3, 3] / current_v

        e = np.linalg.inv(np.eye(4) - (self.dt * A) / 2)  # np.linalg.inv()是矩阵求逆
        disA = e @ (np.eye(4) + (self.dt * A) / 2)

        disB = self.initB * self.dt

        return disA, disB

    def feedforward(self, current_se2state: SE2State, target_se2state: SE2State, K):
        vx = current_se2state.v

        L = self.veh_cfg.wheel_base
        curv = np.tan(target_se2state.delta) / L

        L = self.veh_cfg.wheel_base
        cf = self.veh_cfg.cf_alpha
        cr = self.veh_cfg.cr_alpha
        mass = self.veh_cfg.mass
        lf = self.veh_cfg.lf
        lr = self.veh_cfg.lr

        delta_ff = curv * (
            lf
            + lr
            - lr * K[0, 2]
            - ((mass * vx * vx) / L) * (lr / cf + lf / cr * K[0, 2] - lr / cf)
        )

        # delta_ff = delta_ff * np.pi / 180
        return delta_ff

    def action_online(self, current_se2state: SE2State, target_se2state: SE2State):
        predict_se2state = self.predict(current_se2state)
        # vx = predict_se2state.v
        vx = target_se2state.v
        if abs(vx) < 5 * 1e-3:
            K = np.zeros((1, 4))
        else:
            A, B = self.get_AB_matrix(vx)
            K = dlqr(A, B, self.Q, self.R)

        delta_ff = self.feedforward(predict_se2state, target_se2state, K)

        e = self.error_state(predict_se2state, target_se2state)

        delta = -K @ e + delta_ff

        delta = np.clip(
            delta,
            -self.veh_cfg.max_front_wheel_angle,
            self.veh_cfg.max_front_wheel_angle,
        )

        delta = self.T * delta + (1 - self.T) * self.last_delta
        self.last_delta = delta
        return float(delta)

    def action_lookup(self, current_se2state: SE2State, target_se2state: SE2State):
        return self.action_online(current_se2state, target_se2state)

    def predict(self, current_se2state: SE2State, h=5):
        vx = current_se2state.v
        heading = current_se2state.heading
        vy = vx * np.tan(heading)
        delta = current_se2state.delta

        dt = self.dt * h
        dx = vx * np.cos(heading) * dt - vy * np.sin(heading) * dt
        dy = vy * np.cos(heading) * dt + vx * np.sin(heading) * dt
        dheading = vx / self.veh_cfg.wheel_base * np.tan(delta) * dt

        next_x = current_se2state.x + dx
        next_y = current_se2state.y + dy
        next_heading = current_se2state.heading + dheading

        predict_se2state = SE2State(next_x, next_y, next_heading)
        predict_se2state.v = vx
        predict_se2state.a = current_se2state.a
        predict_se2state.delta = delta

        return predict_se2state

    def error_state(self, current_se2state: SE2State, target_se2state: SE2State):
        d_err = np.array(
            [
                current_se2state.x - target_se2state.x,
                current_se2state.y - target_se2state.y,
            ]
        )

        nor = np.array(
            [-np.sin(target_se2state.heading), np.cos(target_se2state.heading)]
        )

        eheading = (target_se2state.so2 - current_se2state.so2).heading
        v = current_se2state.v
        delta = current_se2state.delta

        ed = d_err @ nor.T
        ed_dot = v * np.sin(eheading)

        L = self.veh_cfg.wheel_base
        v_ref = target_se2state.v
        delta_ref = target_se2state.delta

        eheading_dot = -v_ref * np.tan(delta_ref) / L + v * np.tan(delta) / L

        return np.array([ed, ed_dot, eheading, eheading_dot])


"""
LQR MPC.
"""
import casadi as ca


class OptimalController:
    def __init__(self, veh_cfg=VehicleConfig(), ctrl_cfg=ControlConfig()) -> None:
        self.ctrl_cfg = ctrl_cfg
        self.veh_cfg = veh_cfg
        self.dt = ctrl_cfg.dt
        self.Q = ctrl_cfg.Q
        self.R = ctrl_cfg.R

        self.T = 0.8
        self.last_action = np.array([0, 0], dtype=np.float32)

        self.action_rhs = (
            np.array(
                [
                    self.veh_cfg.max_acc,
                    self.veh_cfg.max_front_wheel_angle,
                    self.veh_cfg.max_acc,
                    self.veh_cfg.max_front_wheel_angle,
                ]
            ).T
            * 5
        )

        self.action_lhs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

        self.runge_kutta = None
        self.build_model()

    def action_point(self, current_se2state: SE2State, target_se2state: SE2State):
        A, B = get_AB_matrix(target_se2state, self.veh_cfg, self.ctrl_cfg)
        K = dlqr(A, B, self.Q, self.R)
        e = self.error_state(current_se2state, target_se2state)

        ff = np.array([target_se2state.a, target_se2state.delta])
        action = -K @ e + ff

        return self.parse_action(action)

    def action_finite_horizon_lqr(
        self,
        current_se2state: SE2State,
        target_se2state_list: List[SE2State],
    ):
        N = len(target_se2state_list)
        Q = self.Q
        R = self.R

        P = Q

        for k in range(N - 1, -1, -1):
            A, B = get_AB_matrix(target_se2state_list[k], self.veh_cfg, self.ctrl_cfg)
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)

        target_se2state = target_se2state_list[0]

        e = self.error_state(current_se2state, target_se2state)
        ff = np.array([target_se2state.a, target_se2state.delta])
        action = -K @ e + ff

        return self.parse_action(action)

    def action_linear_mpc(
        self, current_se2state: SE2State, target_se2state_list: List[SE2State]
    ):
        Nh = len(target_se2state_list)

        Q = self.Q
        R = self.R

        X = cp.Variable((4, Nh))
        U = cp.Variable((2, Nh - 1))

        cost = 0.0

        for i in range(Nh - 1):
            ref_state, ref_action = get_ndarray_state_from_se2state(
                target_se2state_list[i]
            )
            cost += cp.quad_form(X[:, i] - ref_state, Q)
            cost += cp.quad_form(U[:, i] - ref_action, R)

        end_state, end_action = get_ndarray_state_from_se2state(
            target_se2state_list[-1]
        )
        cost += cp.quad_form(X[:, -1] - end_state, Q)

        current_state, current_action = get_ndarray_state_from_se2state(
            current_se2state
        )
        constraints = []
        constraints.append(X[:, 0] == current_state)
        constraints.append(X[:, -1] == end_state)

        # dynamics constraints
        for k in range(Nh - 1):
            A, B = get_AB_matrix(target_se2state_list[k], self.veh_cfg, self.ctrl_cfg)
            ref_state, ref_action = get_ndarray_state_from_se2state(
                target_se2state_list[k]
            )

            next_ref_state, next_ref_action = get_ndarray_state_from_se2state(
                target_se2state_list[k + 1]
            )
            constraints.append(
                X[:, k + 1] - next_ref_state
                == A @ (X[:, k] - ref_state) + B @ (U[:, k] - ref_action)
            )

            # constraints.append(self.action_lhs @ U[:, i] <= 2 *self.action_rhs)

        obj = cp.Minimize(cost)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status == cp.OPTIMAL_INACCURATE:
            print("OPTIMAL_INACCURATE")
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            action = U.value[:, 0]
            return self.parse_action(action)

        else:
            return self.last_action

    def build_model(self):
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        heading = ca.SX.sym("heading")
        v = ca.SX.sym("v")

        state = ca.vertcat(x, y, heading, v)

        a = ca.SX.sym("a")
        delta = ca.SX.sym("delta")

        action = ca.vertcat(a, delta)

        xdot = v * ca.cos(heading)
        ydot = v * ca.sin(heading)
        vdot = a

        headingdot = v / self.veh_cfg.wheel_base * ca.tan(delta)
        statedot = ca.vertcat(xdot, ydot, headingdot, vdot)

        f = ca.Function(
            "f", [state, action], [statedot], ["state", "action"], ["statedot"]
        )

        state = ca.SX.sym("state", 4)
        action = ca.SX.sym("action", 2)
        dt = ca.SX.sym("dt", 1)

        k1 = f(state=state, action=action)["statedot"]
        k2 = f(state=state + dt * k1, action=action)["statedot"]
        next_state = state + dt / 2 * (k1 + k2)

        self.runge_kutta = ca.Function(
            "runge_kutta",
            [state, action, dt],
            [next_state],
        )

    def action_nonlinear_mpc(
        self, current_se2state: SE2State, target_se2state_list: List[SE2State]
    ):
        N = len(target_se2state_list)

        n_states = 4
        n_actions = 2
        X = ca.SX.sym("X", n_states, N)
        U = ca.SX.sym("U", n_actions, N - 1)

        X0, U0 = [], []

        for se2state in target_se2state_list:
            X0 += [[se2state.x, se2state.y, se2state.heading, se2state.v]]

        for i in range(N - 1):
            U0 += [[target_se2state_list[i].a, target_se2state_list[i].delta]]

        variables, lbx, ubx = [], [], []
        for i in range(N):
            variables += [X[:, i]]
            lbx += [-100, -100, -2 * ca.pi, -self.veh_cfg.max_v]
            ubx += [100, 100, 2 * ca.pi, self.veh_cfg.max_v]

        for i in range(N - 1):
            variables += [U[:, i]]
            lbx += [-self.veh_cfg.max_acc, -self.veh_cfg.max_front_wheel_angle]
            ubx += [self.veh_cfg.max_acc, self.veh_cfg.max_front_wheel_angle]

        R = ca.SX(self.R)
        Q = ca.SX(self.Q)
        obj = 0.0

        for i in range(N - 1):
            state = X[:, i]
            error = state - X0[i]
            action = U[:, i]

            obj += action.T @ R @ action
            obj += error.T @ Q @ error

        constraints, lbg, ubg = [], [], []
        constraints += [X[:, 0] - X0[0]]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

        for i in range(N - 1):
            next_state = self.runge_kutta(X[:, i], U[:, i], self.dt)
            constraints += [X[:, i + 1] - next_state]
            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]

        constraints += [X[:, -1] - X0[N - 1]]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

        nlp_prob = {
            "f": obj,
            "x": ca.vertcat(*variables),
            "g": ca.vertcat(*constraints),
        }
        opts = {"print_time": False, "verbose": False, "ipopt.print_level": 0}
        solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
        init_X0 = X0 + U0
        sol = solver(x0=ca.vertcat(*init_X0), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        solution = sol["x"]

        a_opt = solution[
            n_states * (N) : n_states * N + n_actions * (N - 1) : n_actions
        ]

        delta_opt = solution[
            n_states * (N) + 1 : n_states * N + n_actions * (N - 1) : n_actions
        ]

        action = np.array([float(a_opt[0, 0]), float(delta_opt[0, 0])])

        return self.parse_action(action)

    def error_state(self, current_se2state: SE2State, target_se2state: SE2State):
        ex = current_se2state.x - target_se2state.x
        ey = current_se2state.y - target_se2state.y
        eheading = (target_se2state.so2 - current_se2state.so2).heading
        ev = current_se2state.v - target_se2state.v

        return np.array([ex, ey, eheading, ev], dtype=np.float32)

    def parse_action(self, action: np.ndarray):
        # acc = action[0]
        acc = np.clip(
            action[0],
            -self.veh_cfg.max_acc,
            self.veh_cfg.max_acc,
        )
        # delta = action[1]
        delta = np.clip(
            action[1],
            -self.veh_cfg.max_front_wheel_angle,
            self.veh_cfg.max_front_wheel_angle,
        )

        action = np.array([acc, delta], dtype=np.float32)

        action = self.T * action + (1 - self.T) * self.last_action
        self.last_action = action

        return action


"""
Lateral LQR.
"""


class LatLQRController:
    def __init__(self, veh_cfg=VehicleConfig(), ctrl_cfg=ControlConfig()) -> None:
        self.dt = ctrl_cfg.dt
        self.veh_cfg = veh_cfg
        self.ctrl_cfg = ctrl_cfg
        self.Q = ctrl_cfg.Q
        self.R = ctrl_cfg.R

        self.last_delta = 0
        self.T = 0.8  # 1 / T filter.

        import pickle

        file_name = "lat_lqr_lookuptable.pickle"
        with open(file_name, "rb") as f:
            self.lookup_table = pickle.load(f)

    def action_online(self, current_se2state: SE2State, target_se2state: SE2State):
        v = target_se2state.v
        L = self.veh_cfg.wheel_base

        A = np.array(
            [[1, self.dt, 0, 0], [0, 0, v, 0], [0, 0, 1, self.dt], [0, 0, 0, 0]]
        )
        B = np.array([[0], [0], [v / L], [0]])

        K = dlqr(A, B, self.Q, self.R)
        e = self.error_state(current_se2state, target_se2state)

        delta = -K @ e + target_se2state.delta
        delta = np.clip(
            delta,
            -self.veh_cfg.max_front_wheel_angle,
            self.veh_cfg.max_front_wheel_angle,
        )

        delta = self.T * delta + (1 - self.T) * self.last_delta
        self.last_delta = delta
        return float(delta)

    def action_lookup(self, current_se2state: SE2State, target_se2state: SE2State):
        v = current_se2state.v

        K = lookup(v, self.lookup_table)
        e = self.error_state(current_se2state, target_se2state)

        delta = -K @ e + target_se2state.delta
        delta = np.clip(
            delta,
            -self.veh_cfg.max_front_wheel_angle,
            self.veh_cfg.max_front_wheel_angle,
        )

        delta = self.T * delta + (1 - self.T) * self.last_delta
        self.last_delta = delta
        return float(delta)

    def error_state(self, current_se2state: SE2State, target_se2state: SE2State):
        d_err = np.array(
            [
                current_se2state.x - target_se2state.x,
                current_se2state.y - target_se2state.y,
            ]
        )

        nor = np.array(
            [-np.sin(target_se2state.heading), np.cos(target_se2state.heading)]
        )

        eheading = (target_se2state.so2 - current_se2state.so2).heading
        v = current_se2state.v
        delta = current_se2state.delta

        ed = d_err @ nor.T
        ed_dot = v * np.sin(eheading)

        L = self.veh_cfg.wheel_base
        v_ref = target_se2state.v
        delta_ref = target_se2state.delta

        eheading_dot = -v_ref * np.tan(delta_ref) / L + v * np.tan(delta) / L

        return np.array([ed, ed_dot, eheading, eheading_dot])


"""
Basic PID.
"""


class PosPID:
    def __init__(self, kp=1, ki=0.00, kd=0.00, dt=0.02) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.sum_error = 0
        self.last_error = 0

        self.dt = dt

    def action(self, error):
        ie = self.sum_error * self.dt
        de = (error - self.last_error) / self.dt

        self.sum_error += error
        self.last_error = error

        return (self.kp * error) + (self.kd * de) + (self.ki * ie)


class IncPID:
    def __init__(self, kp=1, ki=0.00, kd=0.00, dt=0.02) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.error_k1 = 0
        self.error_k2 = 0

        self.u = 0

        self.dt = dt

    def action(self, error):
        error_k = error
        self.u += (
            self.kp * (error_k - self.error_k1)
            + self.ki * error_k
            + self.kd * (error_k - 2 * self.error_k1 + self.error_k2)
        ) * self.dt

        self.error_k2 = self.error_k1
        self.error_k1 = error_k

        return self.u


"""
Longitudinal Double PID.
"""


class LonPosPIDController:
    def __init__(self, veh_cfg=VehicleConfig(), ctrl_cfg=ControlConfig()) -> None:
        self.ctrl_cfg = ctrl_cfg
        self.veh_cfg = veh_cfg

        self.speedcontroller = PosPID(
            ctrl_cfg.s_p, ctrl_cfg.s_i, ctrl_cfg.s_d, ctrl_cfg.dt
        )

        self.accelcontroller = PosPID(
            ctrl_cfg.v_p, ctrl_cfg.v_i, ctrl_cfg.v_d, ctrl_cfg.dt
        )

    def action(self, current_se2state: SE2State, target_se2state: SE2State):
        s_err = np.array(
            [
                target_se2state.x - current_se2state.x,
                target_se2state.y - current_se2state.y,
            ]
        )

        tor = np.array(
            [np.cos(target_se2state.heading), np.sin(target_se2state.heading)]
        )

        es = s_err @ tor.T

        uv = self.speedcontroller.action(es) + target_se2state.v
        uv = np.clip(
            uv,
            -self.veh_cfg.max_v,
            self.veh_cfg.max_v,
        )

        ev = target_se2state.v - current_se2state.v + uv

        ua = self.accelcontroller.action(ev) + target_se2state.a

        return np.clip(ua, -self.veh_cfg.max_acc, self.veh_cfg.max_acc)


class LonIncPIDController:
    def __init__(self, veh_cfg=VehicleConfig(), ctrl_cfg=ControlConfig()) -> None:
        self.ctrl_cfg = ctrl_cfg
        self.veh_cfg = veh_cfg
        self.dt = ctrl_cfg.dt

        self.speedcontroller = IncPID(
            ctrl_cfg.s_p, ctrl_cfg.s_i, ctrl_cfg.s_d, ctrl_cfg.dt
        )

        self.accelcontroller = IncPID(
            ctrl_cfg.v_p, ctrl_cfg.v_i, ctrl_cfg.v_d, ctrl_cfg.dt
        )

        self.last_u = 0
        self.T = 0.8

    def action(self, current_se2state: SE2State, target_se2state: SE2State):
        s_err = np.array(
            [
                target_se2state.x - current_se2state.x,
                target_se2state.y - current_se2state.y,
            ]
        )

        tor = np.array(
            [np.cos(target_se2state.heading), np.sin(target_se2state.heading)]
        )

        es_k = s_err @ tor.T

        uv = self.speedcontroller.action(es_k)
        uv = np.clip(
            uv,
            -self.veh_cfg.max_v,
            self.veh_cfg.max_v,
        )

        ev_k = target_se2state.v - current_se2state.v + uv

        ua = self.accelcontroller.action(ev_k) + target_se2state.a

        ua = self.T * ua + (1 - self.T) * self.last_u
        ua = np.clip(ua, -self.veh_cfg.max_acc, self.veh_cfg.max_acc)

        self.last_u = ua

        return ua


class TrackingControllerBase:
    def __init__(
        self,
        reference_trajectory: List[SE2State] = None,
        veh_cfg=VehicleConfig(),
        ctrl_cfg=ControlConfig(),
    ) -> None:
        if reference_trajectory is None:
            raise ValueError("TrackingController init with a None trajectory.")

        self.reference_trajectory = deepcopy(reference_trajectory)

    def action(
        self,
        current_se2state: SE2State,
    ):
        pass

    def emergency_stop(self, current_se2state: SE2State):
        pass

    def get_target(self, current_se2state):
        return get_closest_time_states(self.reference_trajectory, current_se2state)


class LatLQRLonPIDController(TrackingControllerBase):
    def __init__(
        self,
        reference_trajectory: List[SE2State] = None,
        veh_cfg=VehicleConfig(),
        ctrl_cfg=ControlConfig(),
    ) -> None:
        super().__init__(reference_trajectory, veh_cfg, ctrl_cfg)
        self.lon_controller = LonIncPIDController(veh_cfg, ctrl_cfg)
        # self.lon_controller = LonPosPIDController(veh_cfg, ctrl_cfg)
        self.lat_controller = LatLQRController(veh_cfg, ctrl_cfg)
        # self.lat_controller = DynamicsLatLQRController(veh_cfg, ctrl_cfg)

    def action(self, current_se2state: SE2State):
        super().action(current_se2state)

        target_se2state = self.get_target(current_se2state)
        accel = self.lon_controller.action(current_se2state, target_se2state)
        delta = self.lat_controller.action_lookup(current_se2state, target_se2state)

        return np.array([accel, delta], dtype=np.float32)

    def get_target(self, current_se2state):
        super().get_target(current_se2state)

        return get_closest_time_states(self.reference_trajectory, current_se2state)


class LatPIDLonSpeedPIDController(TrackingControllerBase):
    def __init__(
        self,
        reference_trajectory: List[SE2State] = None,
        veh_cfg=VehicleConfig(),
        ctrl_cfg=ControlConfig(),
    ) -> None:
        super().__init__(reference_trajectory, veh_cfg, ctrl_cfg)
        self.lon_controller = IncPID(
            ctrl_cfg.v_p, ctrl_cfg.v_i, ctrl_cfg.v_d, ctrl_cfg.dt
        )

        self.lat_controller = IncPID(
            ctrl_cfg.h_p, ctrl_cfg.h_i, ctrl_cfg.h_d, ctrl_cfg.dt
        )

        self.look_ahead = ctrl_cfg.horizon

    def action(self, current_se2state: SE2State):
        super().action(current_se2state)

        target_se2state = self.get_target(current_se2state)
        error_velocity = target_se2state.v - current_se2state.v
        accel = self.lon_controller.action(error_velocity)

        # Get the ego's location and forward vector

        error_heading = (current_se2state.so2 - target_se2state.so2).heading

        delta = self.lat_controller.action(error_heading)
        return np.array([accel, delta], dtype=np.float32)

    def get_target(self, current_se2state):
        # super().get_target(current_se2state)

        return get_minimum_distance_states(
            self.reference_trajectory, current_se2state, self.look_ahead
        )


class OptiTrackingController(TrackingControllerBase):
    def __init__(
        self,
        reference_trajectory: List[SE2State] = None,
        veh_cfg=VehicleConfig(),
        ctrl_cfg=ControlConfig(),
        controller_type="Finite_LQR",  # "Finite_LQR" " MPC"
        horizon=50,
    ) -> None:
        super().__init__(reference_trajectory, veh_cfg, ctrl_cfg)

        self.controller = OptimalController(veh_cfg, ctrl_cfg)
        self.controller_type = controller_type
        self.horizon = horizon

    def action(self, current_se2state: SE2State):
        super().action(current_se2state)

        if self.controller_type == "Finite_LQR":
            target_se2state_list = self.get_target_trajectory(current_se2state)
            action = self.controller.action_finite_horizon_lqr(
                current_se2state, target_se2state_list
            )

        elif self.controller_type == "MPC":
            target_se2state_list = self.get_target_trajectory(current_se2state)
            action = self.controller.action_nonlinear_mpc(
                current_se2state, target_se2state_list
            )

        elif self.controller_type == "LQR":
            target_se2state = self.get_target(current_se2state)
            action = self.controller.action_point(current_se2state, target_se2state)

        else:
            raise ValueError("wrong controller type.")

        return action

    def get_target(self, current_se2state: SE2State):
        super().get_target(current_se2state)

        return get_closest_time_states(self.reference_trajectory, current_se2state)

    def get_target_trajectory(self, current_se2state: SE2State):
        super().get_target(current_se2state)
        return get_closest_time_trajectory(
            self.reference_trajectory, current_se2state, self.horizon
        )
