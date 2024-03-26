import numpy as np

# import geometry
from CarlaAutoParking.others.geometry import (
    Point,
    Polygon,
    move_vertexes_array,
    ndarray_to_vertexlist,
    point_in_polygon,
)
from config import ParkingConfig, VehicleConfig
from CarlaAutoParking.others.se2state import SE2State, SE2
from typing import List, Tuple

from CarlaAutoParking.gridmap.gridmap import GridMap, collision


def parkingse2_to_parking_vertexes(
    parkingse2: SE2, parking_cfg: ParkingConfig = ParkingConfig()
):
    parking_array = parking_cfg.parking_array
    rot_angle = parkingse2.so2.heading
    xy_offset = np.array([[parkingse2.x], [parkingse2.y]])
    parking_array = move_vertexes_array(parking_array, rot_angle, xy_offset)
    parking_vertexes = ndarray_to_vertexlist(parking_array)

    return parking_vertexes


def is_parallel_parking_polygon(
    parking_polygon: Polygon,
    obstalce_polygon_list: List[Polygon],
    parking_cfg=ParkingConfig(),
):
    center, heading = get_parking_polygon_center_heading(parking_polygon)
    parking_length = parking_cfg.parking_length
    parking_width = parking_cfg.parking_width
    center_x, center_y = center[0, 0], center[1, 0]

    x = center_x + parking_length * np.cos(heading)
    y = center_y + parking_length * np.sin(heading)
    for obstacle_polygon in obstalce_polygon_list:
        if point_in_polygon((x, y), obstacle_polygon):
            return True

    x = center_x + parking_length * np.cos(-np.pi + heading)
    y = center_y + parking_length * np.sin(-np.pi + heading)
    for obstacle_polygon in obstalce_polygon_list:
        if point_in_polygon((x, y), obstacle_polygon):
            return True

    x = center_x + parking_width * np.sin(heading)
    y = center_y + parking_width * np.cos(heading)
    for obstacle_polygon in obstalce_polygon_list:
        if point_in_polygon((x, y), obstacle_polygon):
            return False

    x = center_x + parking_width * np.sin(-np.pi + heading)
    y = center_y + parking_width * np.cos(-np.pi + heading)
    for obstacle_polygon in obstalce_polygon_list:
        if point_in_polygon((x, y), obstacle_polygon):
            return False

    return False


def generate_obstacle_and_parking_polygon(parking_cfg=ParkingConfig()):
    """
    ### Generate obstacle and parking polygon.

    ---
    Return: ( parallel parking (vertexes), T parking task (vertexes) )
    """

    (
        obstacle_vertexes_list,
        parking_vertexes_list,
    ) = generate_obstacle_and_parking_vertexes()

    obstacle_polygon_list = []
    parking_polygon_list = []

    for obstacle_vertexes in obstacle_vertexes_list:
        obstacle_polygon_list += [Polygon(obstacle_vertexes)]

    for parking_vertexes in parking_vertexes_list:
        parking_polygon_list += [Polygon(parking_vertexes)]

    return obstacle_polygon_list, parking_polygon_list


def generate_obstacle_and_parking_vertexes(parking_cfg=ParkingConfig()):
    """
    ### Generate obstacle and parking vertexes.

    ---
    Return: ( parallel parking (vertexes), T parking task (vertexes) )
    """

    obstacle_vertexes_list = []
    parking_vertexes_list = []
    """
    Parking space param.
    """
    parking_width = parking_cfg.parking_width
    parking_length = parking_cfg.parking_length
    parking_array = parking_cfg.parking_array

    """
    Parallel Parking
    """
    x_start, y_start = 12, 19
    xy_offset = np.array([[x_start - parking_length], [y_start]])
    neighbor_array = move_vertexes_array(parking_array, 0, xy_offset)
    v_obstacles = ndarray_to_vertexlist(neighbor_array)
    obstacle_vertexes_list.append(v_obstacles)

    xy_offset = np.array([[x_start + parking_length], [y_start]])
    neighbor_array = move_vertexes_array(parking_array, np.pi, xy_offset)
    v_obstacles = ndarray_to_vertexlist(neighbor_array)
    obstacle_vertexes_list.append(v_obstacles)

    xy_offset = np.array([[x_start], [y_start]])
    neighbor_array = move_vertexes_array(parking_array, 0, xy_offset)
    goal_parking_vertexes = ndarray_to_vertexlist(neighbor_array)
    parking_vertexes_list.append(goal_parking_vertexes)

    """
    T Parking
    """
    x_start, y_start = 11, 6
    xy_offset = np.array([[x_start], [y_start]])
    for i in range(3):
        xy_offset = np.array([[x_start + (i + 1) * parking_width], [y_start]])
        neighbor_array = move_vertexes_array(parking_array, np.pi / 2, xy_offset)
        v_obstacles = ndarray_to_vertexlist(neighbor_array)
        obstacle_vertexes_list.append(v_obstacles)

        xy_offset = np.array([[x_start - (i + 1) * parking_width], [y_start]])
        neighbor_array = move_vertexes_array(parking_array, np.pi / 2, xy_offset)
        v_obstacles = ndarray_to_vertexlist(neighbor_array)

        obstacle_vertexes_list.append(v_obstacles)

    xy_offset = np.array([[x_start], [y_start]])
    neighbor_array = move_vertexes_array(parking_array, np.pi / 2, xy_offset)
    goal_parking_vertexes = ndarray_to_vertexlist(neighbor_array)
    parking_vertexes_list.append(goal_parking_vertexes)

    return obstacle_vertexes_list, parking_vertexes_list


def generate_goal_states_from_parking_rectpolygon(
    parking_polygon: Polygon,
    parking_cfg=ParkingConfig(),
) -> List[SE2State]:
    """
    input parking_space_vertexes only one rectangle polygon.
    TODO: Rect(Polygon): only four vertexes.
    """

    if len(parking_polygon.lines) != 4:
        raise ValueError("error, please give a parking rectangle.")

    parking_length = parking_cfg.parking_length
    mid_point, theta = get_parking_polygon_center_heading(parking_polygon)

    goal_state_list = []
    """
    TODO Model3 L:977mm, here we set 1.2, can rewrite at VehicleConfig.
    """
    parking_offset1 = np.array([[parking_length / 2 - 1.5], [0]])
    init_position = move_vertexes_array(parking_offset1, theta, mid_point)
    goal_state_list += [
        SE2State(init_position[0, 0], init_position[1, 0], theta + np.pi)
    ]

    parking_offset2 = -np.array([[parking_length / 2 - 1.5], [0]])
    init_position = move_vertexes_array(parking_offset2, theta, mid_point)

    goal_state_list += [SE2State(init_position[0, 0], init_position[1, 0], theta)]

    return goal_state_list


def get_parking_polygon_center_heading(
    parking_polygon: Polygon, parking_cfg=ParkingConfig()
):
    if len(parking_polygon.lines) != 4:
        raise ValueError("error, please give a parking rectangle.")

    parking_length = parking_cfg.parking_length
    candidate_headingvec_with_length = []
    for line in parking_polygon.lines:
        vec = np.array(line[1]) - np.array(line[0])
        vec_length = np.linalg.norm(vec)
        if vec_length >= parking_length - 1e-2:
            candidate_headingvec_with_length += [(vec, vec_length)]

    if len(candidate_headingvec_with_length) < 1:
        raise ValueError("error, the parking rectangle can't parking a vehicle.")

    vec = max(
        candidate_headingvec_with_length, key=lambda vec_with_length: vec_with_length[1]
    )[0]

    theta = np.arctan2(vec[1], vec[0])
    vertexes_array = np.array(parking_polygon.vertexes).T
    center_x, center_y = np.mean(vertexes_array, axis=1)

    mid_point = np.array([[center_x], [center_y]])

    return mid_point, theta


def generate_random_start_state(gridmap: GridMap):
    while True:
        x = np.random.random() * gridmap.world_width
        y = np.random.random() * gridmap.world_height
        t = np.random.random() * np.pi
        start_state = SE2State(x, y, t)
        # vehicle_vertics = generate_vehicle_vertexes(start_state)
        # utils.plot_polygon_vertexes(vehicle_vertics, linetype="--r")
        # plt.draw()
        # plt.pause(0.1)
        if not collision(gridmap, start_state):
            return start_state


from copy import deepcopy


def sort_parking_polygon(
    parking_polygon_list: List[Polygon],
    current_state: SE2State,
    veh_cfg=VehicleConfig(),
):
    def greater(lhs: Polygon, rhs: Polygon, current_xy: np.ndarray, R=6):
        lhs = abs(np.linalg.norm(np.array(lhs.center) - current_xy) - R)
        rhs = abs(np.linalg.norm(np.array(rhs.center) - current_xy) - R)

        return lhs - rhs

    offset = 1.38

    heading = current_state.heading
    x = current_state.se2.x + offset * np.cos(heading)
    y = current_state.se2.y + offset * np.sin(heading)

    current_xy = np.array([x, y])

    R = veh_cfg.min_radius
    # sort.
    n = len(parking_polygon_list)
    if n < 1:
        raise ValueError("goal state list length < 1")

    if n == 1:
        return parking_polygon_list[0]

    for i in range(n):
        for j in range(n - 1 - i):
            if greater(
                parking_polygon_list[j], parking_polygon_list[j + 1], current_xy, R
            ):
                temp = deepcopy(parking_polygon_list[j])
                parking_polygon_list[j] = deepcopy(parking_polygon_list[j + 1])
                parking_polygon_list[j + 1] = temp

    return parking_polygon_list


def sort_selected_states(goal_state_list: List[SE2State], select_point: Point):
    def greater(lhs: SE2State, rhs: SE2State, select_point: Point):
        lhs_dx = lhs.x - select_point[0]
        lhs_dy = lhs.y - select_point[1]
        lhs_ds = lhs_dx * lhs_dx + lhs_dy * lhs_dy

        rhs_dx = rhs.x - select_point[0]
        rhs_dy = rhs.y - select_point[1]

        rhs_ds = rhs_dx * rhs_dx + rhs_dy * rhs_dy

        return lhs_ds > rhs_ds

    # sort.
    n = len(goal_state_list)
    if n < 1:
        raise ValueError("goal state list length < 1")

    if n == 1:
        return goal_state_list

    for i in range(n):
        for j in range(n - 1 - i):
            if greater(goal_state_list[j], goal_state_list[j + 1], select_point):
                temp = goal_state_list[j]
                goal_state_list[j] = goal_state_list[j + 1]
                goal_state_list[j + 1] = temp

    return goal_state_list


def sort_goal_states(goal_state_list: List[SE2State], start_state: SE2State):
    def greater(lhs: SE2State, rhs: SE2State, start: SE2State):
        lhs_dx = lhs.x - start.x
        lhs_dy = lhs.y - start.y
        lhs_dh = abs((lhs.so2 - start.so2).heading)

        lhs_ds = 1 / np.sqrt(lhs_dx * lhs_dx + lhs_dy * lhs_dy)
        # lhs_dh = abs(lhs_dse2.so2.heading)

        lhs_dh = lhs_dh / (0.5 * np.pi)
        lhs_d = lhs_dh + lhs_ds

        rhs_dx = rhs.x - start.x
        rhs_dy = rhs.y - start.y

        rhs_dh = abs((rhs.so2 - start.so2).heading)

        rhs_ds = 1 / np.sqrt(rhs_dx * rhs_dx + rhs_dy * rhs_dy)
        # rhs_dh = abs(rhs_dse2.so2.heading)
        rhs_dh = rhs_dh / (0.5 * np.pi)
        rhs_d = rhs_dh + rhs_ds

        return lhs_d > rhs_d

    # sort.
    n = len(goal_state_list)
    if n < 1:
        raise ValueError("goal state list length < 1")

    if n == 1:
        return goal_state_list

    for i in range(n):
        for j in range(n - 1 - i):
            if greater(goal_state_list[j], goal_state_list[j + 1], start_state):
                temp = goal_state_list[j]
                goal_state_list[j] = goal_state_list[j + 1]
                goal_state_list[j + 1] = temp

    return goal_state_list


# def generate_random_obstacle_vertexes(grid_map: GridMap, obstacle_num:int=5):
#     import random

#     obstacle_vertexes_list = []
#     while len(obstacle_vertexes_list) < obstacle_num:
#         obstacle_vertexes = [random.randint(0, grid_map.world_width), \
#                              random.randint(0, grid_map.world_height) \
#                             for _ in range(random.randint(3, 5))]

#         obstacle_vertexes_list += [obstacle_vertexes]

#     return obstacle_vertexes_list


def test():
    pass


if __name__ == "__main__":
    test()
