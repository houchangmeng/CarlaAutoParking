import numpy as np
from typing import List
from copy import deepcopy
from config import VehicleConfig, ControlConfig
from CarlaAutoParking.others.se2state import SE2State


def dlqr(A, B, Q, R, eps=1e-3):
    P = Q

    K_last = np.ones((1, 4)) * 1e4

    while True:
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
        if np.linalg.norm(K - K_last, np.inf) < eps:
            break
        K_last = K

    return K


def make_kinematics_lat_lqr_table(
    veh_cfg=VehicleConfig(),
    ctrl_cfg=ControlConfig(),
):
    n = int(5 * veh_cfg.max_v / 0.01)
    L = veh_cfg.wheel_base

    Q = ctrl_cfg.Q
    R = ctrl_cfg.R
    dt = ctrl_cfg.dt

    table = []
    for i in range(-n, n + 1, 1):
        v = 0.01 * i
        A = np.array([[1, dt, 0, 0], [0, 0, v, 0], [0, 0, 1, dt], [0, 0, 0, 0]])
        B = np.array([[0], [0], [v / L], [0]])
        K = dlqr(A, B, Q, R)
        v_K = np.hstack([v, K.reshape((4,))])
        table += [v_K]

    table = np.array(table)

    import pickle

    file_name = "lat_lqr_lookuptable.pickle"
    with open(file_name, "wb") as f:
        pickle.dump(table, f)

    print("lat_lqr_table_finished.")


def get_AB_matrix(
    target_se2state: SE2State,
    veh_cfg=VehicleConfig(),
    ctrl_cfg=ControlConfig(),
    discrete_type="midpoint_euler",
):
    L = veh_cfg.wheel_base
    dt = ctrl_cfg.dt

    ref_v = target_se2state.v
    ref_heading = target_se2state.heading
    ref_delta = target_se2state.delta

    A = np.array(
        [
            [0, 0, -ref_v * np.sin(ref_heading), np.cos(ref_heading)],
            [0, 0, ref_v * np.cos(ref_heading), np.sin(ref_heading)],
            [0, 0, 0, np.tan(ref_delta) / L],
            [0, 0, 0, 0],
        ]
    )

    B = np.array(
        [
            [0, 0],
            [0, 0],
            [0, ref_v / (L * np.cos(ref_delta) ** 2)],
            [1, 0],
        ]
    )

    disB = B * dt
    if discrete_type == "forward_euler":
        disA = np.eye(4) + A * dt
    else:
        e = np.linalg.inv(np.eye(4) - (dt * A) / 2)
        disA = e @ (np.eye(4) + (dt * A) / 2)

    return disA, disB


def get_ndarray_state_from_se2state(target_se2state: SE2State):
    ref_x = target_se2state.x
    ref_y = target_se2state.y
    ref_h = target_se2state.heading
    if ref_h < 0:
        ref_h += np.pi * 2
    ref_v = target_se2state.v

    ref_acc = target_se2state.a
    ref_delta = target_se2state.delta

    ref_state = np.array([ref_x, ref_y, ref_h, ref_v], dtype=np.float32)
    ref_action = np.array([ref_acc, ref_delta], dtype=np.float32)

    return ref_state, ref_action


def binary_search_table(v, table):
    n, nx = table.shape  #

    l_ptr = 0
    r_ptr = n - 1

    if table[l_ptr][0] > v or table[r_ptr][0] < v:
        raise ValueError("Velocity value exceed the lqr table limit.")

    while True:
        m_ptr = int(0.5 * (l_ptr + r_ptr))
        if table[m_ptr][0] <= v and table[m_ptr + 1][0] >= v:
            break

        if v > table[m_ptr][0]:
            l_ptr = m_ptr
        if v < table[m_ptr][0]:
            r_ptr = m_ptr

    t = (v - table[m_ptr][0]) / (table[m_ptr + 1][0] - table[m_ptr][0])

    K = (1 - t) * table[m_ptr][1:] + t * table[m_ptr + 1][1:]

    return K


def lookup(v, table):
    n, nx = table.shape  #

    l_ptr = 0
    r_ptr = n - 1

    if table[l_ptr][0] > v or table[r_ptr][0] < v:
        raise ValueError("Velocity value exceed the lqr table limit.")

    min_v = table[0][0]
    dv = abs(table[0][0] - table[1][0])
    m_ptr = int((v - min_v) / dv)

    t = (v - table[m_ptr][0]) / (table[m_ptr + 1][0] - table[m_ptr][0])

    K = (1 - t) * table[m_ptr][1:] + t * table[m_ptr + 1][1:]

    return K


def one_order_filter(y, y_last, T=0.8):
    return T * y + (1 - T) * y_last


def get_minimum_distance_states(
    se2state_list: List[SE2State], current_state: SE2State, look_ahead_point: int = 4
) -> SE2State:
    def less(lhs: SE2State, rhs: SE2State, current: SE2State):
        lhs_dist = np.linalg.norm(lhs.array_state[:2] - current.array_state[:2])
        rhs_dist = np.linalg.norm(rhs.array_state[:2] - current.array_state[:2])
        return abs(lhs_dist) < abs(rhs_dist)

    # sort.
    n = len(se2state_list)

    min_index = 0
    for i in range(n - look_ahead_point):
        if less(se2state_list[i], se2state_list[min_index], current_state):
            min_index = i

    return se2state_list[min_index + look_ahead_point]


def get_closest_time_states(
    se2state_list: List[SE2State], current_state: SE2State, look_ahead_point: int = 4
) -> SE2State:
    def less(lhs: SE2State, rhs: SE2State, current: SE2State):
        lhs_time = lhs.t - current.t
        rhs_time = rhs.t - current.t
        return abs(lhs_time) < abs(rhs_time)

    # sort.
    n = len(se2state_list)

    min_index = 0
    for i in range(n - look_ahead_point):
        if less(se2state_list[i], se2state_list[min_index], current_state):
            min_index = i

    return se2state_list[min_index + look_ahead_point]


def get_minimum_distance_trajectory(
    se2state_list: List[SE2State], current_state: SE2State, predict_length: int = 30
) -> SE2State:
    def less(lhs: SE2State, rhs: SE2State, current: SE2State):
        lhs_dist = np.linalg.norm(lhs.array_state[:2] - current.array_state[:2])
        rhs_dist = np.linalg.norm(rhs.array_state[:2] - current.array_state[:2])
        return abs(lhs_dist) < abs(rhs_dist)

    # sort.
    n = len(se2state_list)

    end_se2state = deepcopy(se2state_list[-1])
    new_se2state_list = se2state_list + [end_se2state] * predict_length
    min_index = 0

    for i in range(n):
        if less(new_se2state_list[i], new_se2state_list[min_index], current_state):
            min_index = i
    min_index += 1
    return new_se2state_list[min_index : min_index + predict_length]


def get_closest_time_trajectory(
    se2state_list: List[SE2State], current_state: SE2State, predict_length: int = 30
) -> SE2State:
    def less(lhs: SE2State, rhs: SE2State, current: SE2State):
        lhs_time = lhs.t - current.t
        rhs_time = rhs.t - current.t
        return abs(lhs_time) < abs(rhs_time)

    # sort.
    n = len(se2state_list)

    end_se2state = deepcopy(se2state_list[-1])
    new_se2state_list = se2state_list + [end_se2state] * predict_length
    min_index = 0

    for i in range(n):
        if less(new_se2state_list[i], new_se2state_list[min_index], current_state):
            min_index = i
    min_index += 2
    return new_se2state_list[min_index : min_index + predict_length]


if __name__ == "__main__":
    make_kinematics_lat_lqr_table()
