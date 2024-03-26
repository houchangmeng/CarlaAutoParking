from config import VehicleConfig, OptimizeConfig
from CarlaAutoParking.gridmap.gridmap import GridMap
from CarlaAutoParking.others.geometry import get_polygon_halfspaces, Polygon
import casadi as ca
from casadi import numpy as canp
from CarlaAutoParking.others.se2state import SE2State
from typing import List


class Optimizer:
    def __init__(self, veh_cfg=VehicleConfig(), opti_cfg=OptimizeConfig()) -> None:
        self.veh_cfg = veh_cfg
        self.opti_cfg = opti_cfg
        self.opts = opti_cfg.solver_opts
        self.offset = veh_cfg.length / 2 - veh_cfg.baselink_to_rear
        self.n_states = 6
        self.n_actions = 2

        self.MU_LIST = []
        self.LAMBDA_list = []
        self.n_dual_variables_list = []  # lambda , mu
        self.constraints = []
        self.lbg = []  #  lbx < x <ubx
        self.ubg = []
        self.lbx = []  #  lbg < g(x) <ubg
        self.ubx = []
        self.max_x = 99999
        self.max_y = 99999
        self.min_x = -99999
        self.min_y = -99999

        self.variables = []
        self.N = -1
        self.x0 = []
        self.obstacles: List[Polygon] = []
        self.G = ca.DM(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        )
        self.g = ca.DM(
            [
                [veh_cfg.length / 2],
                [veh_cfg.width / 2],
                [veh_cfg.length / 2],
                [veh_cfg.width / 2],
            ]
        )
        self.DT = veh_cfg.T
        self.Q = ca.SX(opti_cfg.Q)
        self.R = ca.SX(opti_cfg.R)
        self.obj = 0
        self.se2trajectory = []

    def initialize(
        self,
        init_se2guess: List[SE2State],
        obstacles_list: List[Polygon],
        gridmap: GridMap,
    ):
        self.N = len(init_se2guess)
        if self.N < 5:
            raise ValueError("init trajectory too short.")

        self.DT = abs(init_se2guess[1].t - init_se2guess[0].t)

        self.obstacles = obstacles_list
        for obstacle in obstacles_list:
            self.n_dual_variables_list += [len(obstacle.vertexes)]

        for se2state in init_se2guess:
            self.x0 += [
                [
                    se2state.x,
                    se2state.y,
                    se2state.heading,
                    se2state.v,
                    se2state.a,
                    se2state.delta,
                ]
            ]
        self.x0[0][3:] = [0, 0, 0]
        # self.x0 += [[0] * (self.n_actions * (self.N - 1))] zeros init inputs.
        for i in range(self.N - 1):
            self.x0 += [[init_se2guess[i].jerk], [init_se2guess[i].delta_dot]]

        for i in range(self.N):
            for n_dual_variables in self.n_dual_variables_list:
                self.x0 += [[0.1] * (n_dual_variables * 2)]  # 2 for lambda and MU

        self.max_x = gridmap.max_x
        self.max_y = gridmap.max_y
        self.min_x = gridmap.min_x
        self.min_y = gridmap.min_y

        # self.x0 += [[0.1] * (self.N - 1)] # for optimize t.

    def build_model(self):
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        heading = ca.SX.sym("heading")
        v = ca.SX.sym("v")
        a = ca.SX.sym("a")
        delta = ca.SX.sym("delta")
        state = ca.vertcat(x, y, heading, v, a, delta)

        jerk = ca.SX.sym("j")
        delta_dot = ca.SX.sym("delta_dot")

        action = ca.vertcat(jerk, delta_dot)

        # beta = ca.arctan(self.veh_cfg.lf * delta / self.veh_cfg.wheel_base)
        # xdot = v * ca.cos(heading + beta)
        # ydot = v * ca.sin(heading + beta)
        # vdot = a
        # adot = jerk
        # headingdot = v * ca.cos(beta) / self.veh_cfg.wheel_base * ca.tan(delta)
        # deltadot = delta_dot

        xdot = v * ca.cos(heading)
        ydot = v * ca.sin(heading)
        vdot = a
        adot = jerk
        headingdot = v / self.veh_cfg.wheel_base * ca.tan(delta)
        deltadot = delta_dot

        statedot = ca.vertcat(xdot, ydot, headingdot, vdot, adot, deltadot)

        self.f = ca.Function(
            "f", [state, action], [statedot], ["state", "action"], ["statedot"]
        )

        state = ca.SX.sym("state", 6)
        action = ca.SX.sym("action", 2)
        dt = ca.SX.sym("dt", 1)

        k1 = self.f(state=state, action=action)["statedot"]
        k2 = self.f(state=state + dt * k1, action=action)["statedot"]
        next_state = state + dt / 2 * (k1 + k2)

        self.runge_kutta = ca.Function(
            "runge_kutta",
            [state, action, dt],
            [next_state],
        )

    def generate_variable(self):
        self.X = ca.SX.sym("X", self.n_states, self.N)
        self.U = ca.SX.sym("U", self.n_actions, self.N - 1)
        # self.DT = ca.SX.sym("DT", self.N - 1)

        for i in range(self.N):
            self.variables += [self.X[:, i]]
            self.lbx += [
                self.min_x,
                self.min_y,
                -ca.pi,
                -self.veh_cfg.max_v,
                -self.veh_cfg.max_acc,
                -self.veh_cfg.max_front_wheel_angle,
            ]
            self.ubx += [
                self.max_x,
                self.max_y,
                ca.pi,
                self.veh_cfg.max_v,
                self.veh_cfg.max_acc,
                self.veh_cfg.max_front_wheel_angle,
            ]

        for i in range(self.N - 1):
            self.variables += [self.U[:, i]]
            self.lbx += [-self.veh_cfg.max_jerk, -self.veh_cfg.max_delta_dot]
            self.ubx += [self.veh_cfg.max_jerk, self.veh_cfg.max_delta_dot]

        for i in range(self.N):
            for n_dual_variables in self.n_dual_variables_list:
                # num_lines = len(obstacle.lines)
                MU = ca.SX.sym("MU", n_dual_variables, 1)
                self.variables += [MU]
                self.MU_LIST += [MU]
                self.lbx += [0.0, 0.0, 0.0, 0.0]
                self.ubx += [1e6, 1e6, 1e6, 1e6]
                LAMBDA = ca.SX.sym("LAMBDA", n_dual_variables, 1)
                self.variables += [LAMBDA]
                self.LAMBDA_list += [LAMBDA]
                self.lbx += [0.0, 0.0, 0.0, 0.0]
                self.ubx += [1e6, 1e6, 1e6, 1e6]

    def generate_objective(self):
        R = ca.SX(self.R)
        Q = ca.SX(self.Q)
        for i in range(self.N - 1):
            state = self.X[:, i]
            ref_state = self.x0[i]
            error = state - ref_state
            action = self.U[:, i]

            self.obj += action.T @ R @ action
            self.obj += error.T @ Q @ error

    def generate_constraints(self):
        self.constraints += [self.X[:, 0] - self.x0[0]]
        self.lbg += [0, 0, 0, 0, 0, 0]
        self.ubg += [0, 0, 0, 0, 0, 0]
        for i in range(self.N - 1):
            next_state = self.runge_kutta(self.X[:, i], self.U[:, i], self.DT)
            self.constraints += [self.X[:, i + 1] - next_state]
            self.lbg += [0, 0, 0, 0, 0, 0]
            self.ubg += [0, 0, 0, 0, 0, 0]
        self.constraints += [self.X[:, -1] - self.x0[self.N - 1]]
        self.lbg += [0, 0, 0, 0, 0, 0]
        self.ubg += [0, 0, 0, 0, 0, 0]

        obstacle_number = len(self.obstacles)
        for i in range(self.N):
            index = 0
            st = self.X[:, i]
            heading = st[2]
            x = st[0]
            y = st[1]
            t = ca.vertcat(
                x + self.offset * ca.cos(heading), y + self.offset * ca.sin(heading)
            )

            rot_line1 = ca.horzcat(ca.cos(heading), -ca.sin(heading))
            rot_line2 = ca.horzcat(ca.sin(heading), ca.cos(heading))
            rot = ca.vertcat(rot_line1, rot_line2)

            for obstacle in self.obstacles:
                A, b = get_polygon_halfspaces(obstacle)
                lamb = ca.vertcat(self.LAMBDA_list[obstacle_number * i + index])
                mu = ca.vertcat(self.MU_LIST[obstacle_number * i + index])
                index += 1
                self.constraints += [ca.dot(A.T @ lamb, A.T @ lamb)]
                self.lbg += [0]
                self.ubg += [1]
                self.constraints += [self.G.T @ mu + (rot.T @ A.T) @ lamb]
                self.lbg += [0, 0]
                self.ubg += [0, 0]
                self.constraints += [(-ca.dot(self.g, mu) + ca.dot(A @ t - b, lamb))]
                self.lbg += [0.001]
                self.ubg += [100000]

    def solve(self):
        if self.N == -1:
            raise ValueError("Give optimizer a initial guess.")
        self.build_model()
        self.generate_variable()
        self.generate_objective()
        self.generate_constraints()

        nlp_prob = {
            "f": self.obj,
            "x": ca.vertcat(*self.variables),
            "g": ca.vertcat(*self.constraints),
        }
        # opts = {"print_time": True, "verbose": False, "ipopt.print_level": 0}
        solver = ca.nlpsol("solver", "ipopt", nlp_prob, self.opti_cfg.solver_opts)
        sol = solver(
            x0=ca.vertcat(*self.x0),
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        solution = sol["x"]

        self.x_opt = solution[0 : self.n_states * (self.N) : self.n_states]
        self.y_opt = solution[1 : self.n_states * (self.N) : self.n_states]
        self.theta_opt = solution[2 : self.n_states * (self.N) : self.n_states]
        self.v_opt = solution[3 : self.n_states * (self.N) : self.n_states]
        self.a_opt = solution[4 : self.n_states * (self.N) : self.n_states]
        self.delta_opt = solution[5 : self.n_states * (self.N) : self.n_states]
        self.j_opt = solution[
            self.n_states * (self.N) : self.n_states * (self.N)
            + self.n_actions * (self.N - 1) : self.n_actions
        ]

        self.deltadot_opt = solution[
            self.n_states * (self.N)
            + 1 : self.n_states * (self.N)
            + self.n_actions * (self.N - 1) : self.n_actions
        ]

    def extract_result(self, current_time: float = 0):
        for i in range(self.N - 1):
            x = float(self.x_opt[i])
            y = float(self.y_opt[i])
            h = float(self.theta_opt[i])
            se2state = SE2State(x, y, h)
            se2state.t = float(self.DT * i) + current_time
            se2state.v = float(self.v_opt[i])
            se2state.a = float(self.a_opt[i])
            se2state.delta = float(self.delta_opt[i])
            se2state.jerk = float(self.j_opt[i])

            se2state.delta_dot = float(self.deltadot_opt[i])
            se2state.curv = (
                float(self.v_opt[i]) / 3 * ca.numpy.tan(self.delta_opt[i])
            ) / se2state.v
            self.se2trajectory += [se2state]

        i = self.N - 1
        x = float(self.x_opt[i])
        y = float(self.y_opt[i])
        h = float(self.theta_opt[i])
        se2state = SE2State(x, y, h)
        se2state.t = float(self.DT * i) + current_time
        se2state.v = float(self.v_opt[i])
        se2state.a = float(self.a_opt[i])
        se2state.delta = float(self.delta_opt[i])
        se2state.jerk = float(0)
        se2state.delta_dot = float(0)
        self.se2trajectory += [se2state]

        if len(self.se2trajectory) < 2:
            raise ValueError("Solve Failed.")

        return self.se2trajectory
