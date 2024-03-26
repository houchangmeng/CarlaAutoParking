import math
import numpy as np
from typing import List


import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from CarlaAutoParking.others.geometry import (
    Polygon,
    PolygonContainer,
    Point,
    Line,
    polygon_intersect_polygon,
    point_in_polygon,
    point_distance_polygon,
)

from CarlaAutoParking.others.se2state import SE2State, generate_vehicle_vertexes
from config import GridMapConfig


class GridMap:  # Obstacle Manager..
    def __init__(
        self, min_x=0, min_y=0, max_x=24, max_y=24, gridmap_cfg=GridMapConfig()
    ):
        self.grid_cfg = gridmap_cfg
        self.bound = gridmap_cfg.bound

        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.world_width = max_x - min_x
        self.world_height = max_y - min_y

        self.map_w = int(self.world_width / gridmap_cfg.xy_resolution)
        self.map_h = int(self.world_height / gridmap_cfg.xy_resolution)
        self.headings = int(math.pi * 2 / gridmap_cfg.heading_resolution)

        self.obstacles = PolygonContainer()

    def add_polygon_obstacle(self, obstacle_polygon: Polygon):
        max_x, max_y = np.max(obstacle_polygon.ndarray, axis=1)
        min_x, min_y = np.min(obstacle_polygon.ndarray, axis=1)

        expansion_flag = True
        if min_x < self.min_x:
            self.min_x = min_x
            expansion_flag = True

        if min_y < self.min_y:
            self.min_y = min_y
            expansion_flag = True

        if max_x > self.max_x:
            self.max_x = max_x
            expansion_flag = True

        if max_y > self.max_y:
            self.max_y = max_y
            expansion_flag = True

        if expansion_flag:
            self.world_width = self.max_x - self.min_x
            self.world_height = self.max_y - self.min_y

            self.map_w = int(self.world_width / self.grid_cfg.xy_resolution)
            self.map_h = int(self.world_height / self.grid_cfg.xy_resolution)

        self.obstacles += obstacle_polygon

    def add_polygon_obstacle_list(self, obstacle_polygon_list: List[Polygon]):
        for polygon in obstacle_polygon_list:
            self.add_polygon_obstacle(polygon)

    def add_vertexes_obstacle_list(self, obstacle_vertexes_list: List[List[Point]]):
        for v_obstacles in obstacle_vertexes_list:
            self.add_vertexes_obstacle(v_obstacles)

    def add_vertexes_obstacle(self, obstacle_vertexes: List[Point]):
        self.obstacles += Polygon(obstacle_vertexes)


def generate_gridmap_from_polygon(
    obstacle_polygon_list: List[Polygon],
    parking_polygon_list: List[Polygon],
    current_se2state: SE2State,
):
    if len(parking_polygon_list) == 0:
        return None

    """
    ### generate grid map from polygon.
    """

    point_varray = np.array(generate_vehicle_vertexes(current_se2state))

    for obstacle in obstacle_polygon_list:
        point_varray = np.vstack([point_varray, np.array(obstacle.vertexes)])

    for parking in parking_polygon_list:
        point_varray = np.vstack([point_varray, np.array(parking.vertexes)])

    min_x, min_y = np.min(point_varray, axis=0) - 0.2
    max_x, max_y = np.max(point_varray, axis=0) + 0.2

    grid = GridMap(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    grid.add_polygon_obstacle_list(obstacle_polygon_list)

    return grid


def generate_gridmap_from_parking_task(
    obstacle_polygon_list: List[Polygon],
    current_se2state: SE2State,
    goal_se2state: SE2State,
):
    if obstacle_polygon_list is None:
        return None

    """
    ### generate grid map from polygon.
    """

    point_varray = np.array(generate_vehicle_vertexes(current_se2state))
    point_varray = np.vstack(
        [point_varray, np.array(generate_vehicle_vertexes(goal_se2state))]
    )

    for obstacle in obstacle_polygon_list:
        point_varray = np.vstack([point_varray, np.array(obstacle.vertexes)])

    min_x, min_y = np.min(point_varray, axis=0) - 0.2
    max_x, max_y = np.max(point_varray, axis=0) + 0.2

    grid = GridMap(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    grid.add_polygon_obstacle_list(obstacle_polygon_list)

    return grid


def outboundary(grid: GridMap, state: SE2State):
    if (
        state.x >= grid.max_x
        or state.x < grid.min_x
        or state.y >= grid.max_y
        or state.y < grid.min_y
    ):
        return True
    return False


def collision(grid: GridMap, state: SE2State):
    if outboundary(grid, state):
        return True

    vehicle_vertices = generate_vehicle_vertexes(state)
    vehicle_array = np.array(vehicle_vertices).T
    maxx = np.max(vehicle_array[0, :])
    maxy = np.max(vehicle_array[1, :])
    minx = np.min(vehicle_array[0, :])
    miny = np.min(vehicle_array[1, :])

    if (
        maxx >= grid.max_x
        or maxy >= grid.max_y
        or minx < grid.min_x
        or miny < grid.min_y
    ):
        return True

    if grid.obstacles is not None:
        vehicle_polygon = Polygon(vehicle_vertices)
        for obstacle_polygon in grid.obstacles:
            if polygon_intersect_polygon(obstacle_polygon, vehicle_polygon):
                return True

    return False


def trajectory_collision(grid: GridMap, trajectory: List[SE2State]):
    for state in trajectory:
        if collision(grid, state):
            return True

    return False


def generate_visited_map_3d(grid: GridMap) -> List[List[List[bool]]]:
    visited_map = [
        [
            [False for _ in range(0, grid.headings + grid.bound)]
            for _ in range(0, grid.map_h + grid.bound)
        ]
        for _ in range(grid.map_w + grid.bound)
    ]

    return visited_map


def generate_visited_map_2d(grid: GridMap) -> List[List[bool]]:
    visited_map = [
        [False for _ in range(0, grid.map_h + grid.bound)]
        for _ in range(0, grid.map_w + grid.bound)
    ]
    return visited_map


def generate_heuristic_map(grid: GridMap) -> List[List[float]]:
    heuristic_map = [
        [200.0 for _ in range(0, grid.map_h + grid.bound)]
        for _ in range(0, grid.map_w + grid.bound)
    ]
    return heuristic_map


def generate_obstacle_grid(grid: GridMap) -> List[List[bool]]:
    girdmap_cfg = grid.grid_cfg
    obstacle_field_map = generate_visited_map_2d(grid)
    obstacle_field_map = np.array(obstacle_field_map)
    obstacle_field_map[0, :] = False
    obstacle_field_map[:, 0] = False
    obstacle_field_map[-1, :] = False
    obstacle_field_map[:, -1] = False

    if grid.obstacles is None:
        return obstacle_field_map

    for x in range(grid.map_w):
        xx = x * girdmap_cfg.xy_resolution + grid.min_x
        for y in range(grid.map_h):
            yy = y * girdmap_cfg.xy_resolution + grid.min_y
            for obstacle_polygon in grid.obstacles:
                if point_in_polygon((xx, yy), obstacle_polygon):
                    obstacle_field_map[x][y] = True

    return obstacle_field_map


def calc_likelihood_field(grid: GridMap, goal_se2state: SE2State):
    goal_x = goal_se2state.x
    goal_y = goal_se2state.y

    girdmap_cfg = grid.grid_cfg
    likelihood_field_map = generate_visited_map_2d(grid)
    likelihood_field_map = np.array(likelihood_field_map) * 0

    max_distance = 1e6
    likelihood_field_map[0, :] = max_distance
    likelihood_field_map[:, 0] = max_distance
    likelihood_field_map[-1, :] = max_distance
    likelihood_field_map[:, -1] = max_distance

    if grid.obstacles is None:
        return likelihood_field_map

    for ix in range(grid.map_w):
        xx = ix * girdmap_cfg.xy_resolution + grid.min_x
        for iy in range(grid.map_h):
            yy = iy * girdmap_cfg.xy_resolution + grid.min_y

            ug = calc_attractive_potential((xx, yy), (goal_x, goal_y))
            uo = calc_repulsive_potential((xx, yy), grid.obstacles)
            uf = ug + uo
            likelihood_field_map[ix][iy] = uf

    return likelihood_field_map


KP = 3
ETA = 100


def calc_attractive_potential(point, goal_point):
    x, y = point
    gx, gy = goal_point
    return 0.5 * KP * math.hypot(x - gx, y - gy)


def calc_repulsive_potential(point, obstacle_polygon_list, rr=1):
    min_dis = 1e6
    for obstacle_polygon in obstacle_polygon_list:
        if point_in_polygon(point, obstacle_polygon):
            return 100

        dis = point_distance_polygon(point, obstacle_polygon)
        min_dis = min(dis, min_dis)

    dq = min_dis

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0
