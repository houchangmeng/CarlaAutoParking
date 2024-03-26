from CarlaAutoParking.others.se2state import SE2State, SE2
from config import VehicleConfig, SearchConfig
from CarlaAutoParking.gridmap.gridmap import GridMap
from typing import List
import numpy as np


def compute_2d_index(gridmap, se2state: SE2State):
    se2state.x_index = int(
        (se2state.x - gridmap.min_x) / gridmap.grid_cfg.xy_resolution
    )
    se2state.y_index = int(
        (se2state.y - gridmap.min_y) / gridmap.grid_cfg.xy_resolution
    )
    se2state.heading_index = int(
        (se2state.heading + np.pi) / gridmap.grid_cfg.heading_resolution
    )


def compute_3d_index(gridmap, se2state: SE2State):
    se2state.x_index = int(
        (se2state.x - gridmap.min_x) / gridmap.grid_cfg.xy_resolution
    )
    se2state.y_index = int(
        (se2state.y - gridmap.min_y) / gridmap.grid_cfg.xy_resolution
    )
    se2state.heading_index = int(
        (se2state.heading + np.pi) / gridmap.grid_cfg.heading_resolution
    )


def update_se2state(state: SE2State, vel, delta, gridmap, vehicle_cfg=VehicleConfig()):
    """
    x_k+1 = x_k + vel * np.cos(heading_k) * T
    y_k+1 = y_k + vel * np.sin(heading_k) * T
    heading_k+1 = heading_k + vel / SE2State.vehicle_cfg.wheel_base * np.tan(delta) * T
    """

    T = vehicle_cfg.T
    dx = vel * np.cos(state.heading) * T
    dy = vel * np.sin(state.heading) * T
    dheading = vel / vehicle_cfg.wheel_base * np.tan(delta) * T

    next_x = state.x + dx
    next_y = state.y + dy
    next_heading = state.heading + dheading
    se2 = SE2(next_x, next_y, next_heading)
    # se2 = state.se2 + SE2(dx, dy, dheading)

    new_se2state = SE2State.from_se2(se2=se2)

    new_se2state.t = state.t + T
    new_se2state.v = vel
    new_se2state.delta = delta

    compute_3d_index(gridmap, new_se2state)

    return new_se2state


def get_next_states(
    state: SE2State,
    grid: GridMap,
    vehicle_cfg=VehicleConfig(),
    search_cfg=SearchConfig(),
):
    """
    ### Generate next SE2.

    rl  4 \  / 1  fl
    r    5 -- 2   f
    rr  6 /  \ 3  fr
    """

    next_states = []
    delta_discrete_num = search_cfg.discrete_delta_num
    vel_discrete_num = search_cfg.discrete_delta_num
    velocity_reso = vehicle_cfg.max_v / delta_discrete_num
    delta_reso = vehicle_cfg.max_front_wheel_angle / vel_discrete_num

    for discre_delta in range(-delta_discrete_num, delta_discrete_num + 1):
        for discrete_vel in range(-vel_discrete_num, vel_discrete_num + 1):
            if discrete_vel == 0:
                continue
            velocity = velocity_reso * discrete_vel
            delta = delta_reso * discre_delta
            new_state = update_se2state(state, velocity, delta, grid, vehicle_cfg)

            # new_state.delta = delta
            # new_state.v = velocity
            # new_state.direction_index = get_direction(velocity, delta)

            next_states += [new_state]

    return next_states


def get_direction(velocity, delta):
    """
    ### Check direction.
    rl  4 \  / 1  fl
    r    5 -- 2   f
    rr  6 /  \ 3  fr
    """
    if velocity > 0:
        if delta > 0:
            direction = 1
        elif delta == 0:
            direction = 2
        else:
            direction = 3
    elif velocity < 0:
        if delta > 0:
            direction = 4
        elif delta == 0:
            direction = 5
        else:
            direction = 6
    else:
        raise ValueError(" Error velocity")

    return direction


def get_next_grid_state(state: SE2State, gridmap: GridMap):
    """
    ### Generate next Grid SE2.
            le
            7
    rl  4 \ | / 1  fl
    r    5 --- 2   f
    rr  6 / | \ 3  fr
            8
            ri
    """
    next_states = []
    gridmap_cfg = gridmap.grid_cfg

    offset_list = [
        [gridmap_cfg.xy_resolution, gridmap_cfg.xy_resolution, np.pi / 4, 1.18],
        [gridmap_cfg.xy_resolution, 0, 0, 1],
        [gridmap_cfg.xy_resolution, -gridmap_cfg.xy_resolution, -np.pi / 4, 1.18],
        [
            -gridmap_cfg.xy_resolution,
            gridmap_cfg.xy_resolution,
            np.pi * 3 / 4,
            1.18,
        ],
        [-gridmap_cfg.xy_resolution, 0, np.pi, 1],
        [
            -gridmap_cfg.xy_resolution,
            -gridmap_cfg.xy_resolution,
            -np.pi * 3 / 4,
            1.18,
        ],
        [0, gridmap_cfg.xy_resolution, np.pi * 2 / 4, 1],
        [0, -gridmap_cfg.xy_resolution, -np.pi * 2 / 4, 1],
    ]

    for offset in offset_list:
        x = state.x + offset[0]
        y = state.y + offset[1]
        heading = state.heading + offset[2]
        new_state = SE2State(x, y, heading)
        compute_3d_index(gridmap, new_state)
        new_state.v = offset[3]  # for cost
        new_state.delta = 0.0  # for cost
        next_states += [new_state]

    return next_states


from copy import deepcopy


def downsample_smooth(
    path: List[SE2State],
    gap: int = 3,
):
    if not path:
        print("no path")
        return []

    ds_path = deepcopy(path[::gap])
    if len(ds_path) < 3:
        return ds_path

    dt = gap * (path[1].t)
    for i in range(1, len(ds_path) - 1):
        v = 0
        d = 0
        for k in range(gap + 2 - 1):
            v += path[i * gap + k].v
            v += path[i * gap - k].v

            d += path[i * gap + k].delta
            d += path[i * gap - k].delta

        ds_path[i].v = v / gap / 2
        ds_path[i].a = (ds_path[i].v - ds_path[i - 1].v) / dt
        ds_path[i].delta = d / gap / 2

    ds_path[-1] = path[-1]
    return ds_path


def upsample_interpolation(
    path: List[SE2State],
    interval: int = 3,
) -> List[SE2State]:
    from scipy import interpolate

    if len(path) < 5 or interval < 1:
        raise ValueError("check path or interval number.")

    T = path[1].t - path[0].t
    t = np.array([0, T])
    delta_t = T / (interval + 1)  # segement num
    t_seq = np.linspace(0, T, interval + 2)  # start + end, 2 waypoints

    us_path = []
    # path = path[::interval]
    for i in range(0, len(path) - 1):
        x = np.array([path[i].x, path[i + 1].x])
        y = np.array([path[i].y, path[i + 1].y])
        heading = np.array([path[i].heading, path[i + 1].heading])
        vel = np.array([path[i].v, path[i + 1].v])
        acc = np.array([path[i].a, path[i + 1].a])
        jerk = np.array([path[i].jerk, path[i + 1].jerk])
        delta = np.array([path[i].delta, path[i + 1].delta])
        delta_dot = np.array([path[i].delta_dot, path[i + 1].delta_dot])

        yy = np.vstack((x, y, heading, vel, acc, jerk, delta, delta_dot))
        interp_func = interpolate.interp1d(t, yy)
        state_sub_seq = interp_func(t_seq)

        for k in range(interval + 2 - 1):
            x, y, heading, v, a, j, delta, ddelta = state_sub_seq[:, k]
            new_state = SE2State(x, y, heading)
            new_state.t = i * T + k * delta_t
            new_state.v = v
            new_state.a = a
            new_state.jerk = j
            new_state.delta = delta
            new_state.delta_dot = ddelta

            us_path += [new_state]

    path[-1].t = us_path[-1].t + delta_t
    us_path += [path[-1]]

    return us_path


def downup_smooth(path: List[SE2State], interval: int = 3):
    ds_path = downsample_smooth(path, interval)
    us_path = upsample_interpolation(ds_path, interval - 1)

    return us_path


def trajectory_add_zero_velocity(path: List[SE2State], insert_num=5):
    if len(path) < 3:
        raise ValueError("check path or interval number.")

    dt = path[1].t - path[0].t
    N = len(path)
    new_path: List[SE2State] = []

    for _ in range(insert_num):
        new_path += [deepcopy(path[0])]

    for i in range(N - 1):
        if (path[i].v * path[i + 1].v) <= 0:
            for _ in range(insert_num):
                new_path += [deepcopy(path[i + 1])]
        else:
            new_path += [path[i + 1]]

    for _ in range(insert_num):
        new_path += [deepcopy(path[-1])]

    new_N = len(new_path)

    for j in range(new_N):
        new_path[j].t = dt * j

    return new_path


def back_track_state(end: SE2State) -> List[SE2State]:
    path = []
    while end.parent is not None:
        path += [end]
        parent = end.parent
        end = deepcopy(parent)
    path += [end]

    return path[::-1]


def back_track_close(
    close_list,
    vehicle_cfg=VehicleConfig(),
):
    if len(close_list) < 1:
        print("empty close list")
        return
    end = close_list[-1]

    path = back_track_state(end)

    N = len(path) - 1
    for i in range(0, N):
        path[i].t = i * vehicle_cfg.T
        # dx = path[i + 1].x - path[i].x
        # dy = path[i + 1].y - path[i].y
        # ds = np.sqrt(dx * dx + dy * dy)
        # dh = (path[i + 1].so2 - path[i].so2).heading
        # path[i].curv = ds / dh
    return path


def back_track_merge(
    merge_state_from_start: SE2State,
    merge_state_from_goal: SE2State,
    vehicle_cfg=VehicleConfig(),
):
    end: SE2State = merge_state_from_start  # take from start
    path1: List[SE2State] = back_track_state(end)

    end: SE2State = merge_state_from_goal  # take from goal
    path2: List[SE2State] = back_track_state(end)

    for se2state in path2:
        se2state.v = -se2state.v

    path2.reverse()

    path = path1 + path2
    for i in range(0, len(path)):
        path[i].t = i * vehicle_cfg.T

    return path


def compute_curvature(x, y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    refer to https://github.com/Pjer-zhang/PJCurvature for detail
    """

    t_a = np.linalg.norm([x[1] - x[0], y[1] - y[0]])
    t_b = np.linalg.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([[1, -t_a, t_a**2], [1, 0, 0], [1, t_b, t_b**2]])

    a = np.matmul(np.linalg.inv(M), x)
    b = np.matmul(np.linalg.inv(M), y)

    kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2.0 + b[1] ** 2.0) ** (1.5)
    return kappa, [b[1], -a[1]] / np.sqrt(a[1] ** 2.0 + b[1] ** 2.0)


import CarlaAutoParking.planner.reeds_sheep as rs
from queue import PriorityQueue
from CarlaAutoParking.gridmap.gridmap import collision


def analystic_expantion(
    current_se2state: SE2State,
    goal_se2state: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
    veh_cfg=VehicleConfig(),
):
    sx, sy, syaw = current_se2state.x, current_se2state.y, current_se2state.heading
    gx, gy, gyaw = goal_se2state.x, goal_se2state.y, goal_se2state.heading

    maxc = np.tan(veh_cfg.max_front_wheel_angle) / veh_cfg.wheel_base
    paths: List[rs.PATH] = rs.calc_all_paths(
        sx,
        sy,
        syaw,
        gx,
        gy,
        gyaw,
        maxc,
        step_size=(veh_cfg.max_v / search_cfg.discrete_velocity_num) * veh_cfg.T,
    )

    if not paths:
        return None
    pq = PriorityQueue()

    for path in paths:
        rs_path_cost = sum(path.lengths)
        pq.put((rs_path_cost, path), block=False)

    while not pq.empty():
        available_path = True
        path = pq.get()[1]
        ind = range(0, len(path.x))

        path_x = [path.x[k] for k in ind]
        path_y = [path.y[k] for k in ind]
        path_heading = [path.yaw[k] for k in ind]

        for i in range(0, len(path.x)):
            x = path_x[i]
            y = path_y[i]
            heading = path_heading[i]
            path_se2state = SE2State(x, y, heading)

            if collision(grid_map, path_se2state):
                available_path = False
                break

        if available_path:
            return path

    return None


def update_path_with_analystic_expantion(
    current_se2state: SE2State,
    goal_se2state: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
    veh_cfg=VehicleConfig(),
):
    path: rs.PATH = analystic_expantion(
        current_se2state, goal_se2state, grid_map, search_cfg, veh_cfg
    )

    if not path:
        return False, None

    N = len(path.x)
    rs_se2state_path: List[SE2State] = []
    for i in range(1, N - 1):
        x = path.x[i]
        y = path.y[i]
        heading = path.yaw[i]
        v = path.directions[i] * veh_cfg.max_v / search_cfg.discrete_velocity_num

        delta = ((path.yaw[i] - path.yaw[i - 1]) * veh_cfg.wheel_base) / v
        se2state = SE2State(x, y, heading)
        se2state.v = v
        se2state.delta = delta
        rs_se2state_path += [se2state]

    return True, rs_se2state_path

    # def analystic_expand(
    #     current_state: SE2State,
    #     goal_state: SE2State,
    #     grid: GridMap,
    #     vehicle_cfg=VehicleConfig(),
    # ):
    #     """
    #     TODO analystic expand trajectory
    #     """

    # def quintic_coefficient(start, end, T):
    #     """Calculate quintic polynomial."""
    #     b = np.array([start, end, 0, 0, 0, 0])
    #     A = np.array(
    #         [
    #             [1, 0, 0, 0, 0, 0],
    #             [1, 1 * T, 1 * T**2, 1 * T**3, 1 * T**4, 1 * T**5],
    #             [0, 1, 0, 0, 0, 0],
    #             [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
    #             [0, 0, 2, 0, 0, 0],
    #             [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
    #         ]
    #     )

    #     p = np.dot(np.linalg.inv(A), b)
    #     return p

    # def linear_intep(start, end, T):
    #     A = np.array([[x1, 1], [x2, 1]])
    #     b = np.array([start, end])
    #     p = np.dot(np.linalg.inv(A), b)
    #     return p

    # def linear_value(p, t):
    #     return p @ np.array([t, 1]).T

    # def quintic_value(p, t):
    #     x = (
    #         p[0]
    #         + p[1] * t
    #         + p[2] * t**2
    #         + p[3] * t**3
    #         + p[4] * t**4
    #         + p[5] * t**5
    #     )
    #     x_dot = (
    #         p[1]
    #         + 2 * p[2] * t
    #         + 3 * p[3] * t**2
    #         + 4 * p[4] * t**3
    #         + 5 * p[5] * t**4
    #     )

    #     return x, x_dot

    # dx = goal_state.x - current_state.x
    # dy = goal_state.y - current_state.y
    # dheading = (goal_state.so2 - current_state.so2).heading

    # distance = np.sqrt(dx * dx + dy * dy)
    # T = int(distance / gridmap_cfg.xy_resolution)
    # Th = int(abs(dheading) / gridmap_cfg.heading_resolution)

    # px = quintic_coefficient(current_state.x, goal_state.x, T)
    # py = quintic_coefficient(current_state.y, goal_state.y, T)
    # ph = quintic_coefficient(current_state.heading, goal_state.heading, T)

    # """
    # TODO : ERROR.
    # """
    # analystic_path = []
    # for t in range(1, int(T)):
    #     next_h, next_hdot = quintic_value(ph, t)
    #     next_x, next_xdot = quintic_value(px, t)
    #     next_y, next_ydot = quintic_value(py, t)

    #     # next_h = linear_value(ph, t)
    #     # dx = next_x - current_state.x
    #     # dy = next_y - current_state.y
    #     # dh = next_h - current_state.heading

    #     # w_vec = np.array([dx, dy, 0])
    #     # v_vec = np.array(
    #     #     [np.cos(current_state.heading), np.sin(current_state.heading), 0]
    #     # )
    #     # wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
    #     # dh = np.arccos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
    #     # cross = np.cross(v_vec, w_vec)
    #     # dh *= np.sign(cross[2])

    #     # se2 = current_state.se2 + SE2(dx, dy, dh)
    #     # se2state = SE2State.from_se2(se2)

    #     se2state = SE2State(next_x, next_y, next_h)

    #     if collision(grid, se2state):
    #         return []

    #     dh = (se2state.so2 - current_state.so2).heading
    #     # if abs(dh) > 0.1:
    #     #     return []
    #     v = np.sqrt(next_xdot**2 + next_ydot**2)
    #     se2state.v = v * np.cos(next_h)

    #     se2state.delta = np.clip(
    #         np.arctan(dh * vehicle_cfg.wheel_base / se2state.v),
    #         -vehicle_cfg.max_front_wheel_angle,
    #         vehicle_cfg.max_front_wheel_angle,
    #     )
    #     analystic_path += [se2state]
    #     current_state = se2state

    # # analystic_path += [goal_state]

    # return analystic_path
    raise NotImplementedError("analystic_expand function has not be implementated.")
