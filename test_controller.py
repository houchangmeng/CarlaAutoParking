import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))


import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import ControlConfig
import CarlaAutoParking.utils.plt_utils as plt_utils
from CarlaAutoParking.utils.parking_utils import generate_obstacle_and_parking_polygon
from CarlaAutoParking.planner.serach_utils import (
    downsample_smooth,
    trajectory_add_zero_velocity,
)

from CarlaAutoParking.others.kinematics import runge_kutta_integration
from CarlaAutoParking.others.se2state import SE2State
from CarlaAutoParking.controller.controller import OptiTrackingController


def simulation_kinematics(se2state: SE2State, action, h=0.02):
    state = se2state.array_state[:4]

    x, y, heading, v = runge_kutta_integration(state, action, h)
    next_se2state: SE2State = SE2State(x, y, heading)
    next_se2state.v = float(v)
    next_se2state.a = float(action[0])
    next_se2state.delta = float(action[1])
    next_se2state.t = float(se2state.t + h)

    return next_se2state


TASK_NUM = 0


def test_controller():
    (
        obstacle_polygon_list,
        _,
    ) = generate_obstacle_and_parking_polygon()
    plt.figure(0, figsize=[8, 8])

    file_name = "se2opti_path" + str(TASK_NUM) + ".pickle"

    with open(file_name, "rb") as f:
        opti_path = pickle.load(f)

    start_state: SE2State = opti_path[0]
    goal_state = opti_path[-1]
    plt_utils.plot_task(obstacle_polygon_list, start_state, goal_state)

    plt_utils.plot_path(opti_path, "opti_path")
    plt.draw()
    plt.pause(0.1)

    zero_path = trajectory_add_zero_velocity(opti_path)

    plt_utils.plot_path(zero_path, "zero_path")
    plt.draw()
    plt.pause(0.1)

    ctrl_cfg = ControlConfig()
    ctrl_cfg.Q = np.diag([10, 10, 100, 10])
    ctrl_cfg.R = np.diag([1, 1])
    # controller = LatLQRLonPIDController(
    #     ctrl_cfg=ctrl_cfg, reference_trajectory=opti_path
    # )
    controller = OptiTrackingController(
        ctrl_cfg=ctrl_cfg,
        reference_trajectory=opti_path,
        controller_type="LQR",
        horizon=10,
    )

    # controller = Controller()
    # controller = LatLonMPCController()
    current_se2state = start_state
    tracking_path = [current_se2state]

    start_time = time.time()

    N = len(opti_path)
    # N = 200
    for i in range(N):
        u = controller.action(current_se2state)
        current_se2state = simulation_kinematics(current_se2state, u)
        tracking_path += [current_se2state]
        if i % 10 == 0:
            print(f"Current step is {i} / {N}")

    print(f"total time is f{time.time() - start_time:10.6f}")
    plt_utils.plot_path(tracking_path, "tracking_path")
    ds_tracking_path = downsample_smooth(tracking_path, 12)
    plt_utils.plot_trajectory_animation(ds_tracking_path)

    plt.figure(1, figsize=[8, 10])
    plt_utils.plot_control(zero_path)
    plt_utils.plot_control(tracking_path)
    plt.draw()
    plt.show()
    print("123")


if __name__ == "__main__":
    test_controller()
    # make_kinematics_lat_lqr_table()
