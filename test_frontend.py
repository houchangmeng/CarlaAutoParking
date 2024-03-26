import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))


import numpy as np
import matplotlib.pyplot as plt
import CarlaAutoParking.utils.plt_utils as plt_utils
import pickle
import time
from CarlaAutoParking.planner.search import (
    Frontend,
    hybrid_a_star_search,
    hybrid_a_star_reverse_search,
    hybrid_a_star_reverse_search_with_analystic_expantion,
    breadth_first_search,
    bidirection_hybrid_a_star_search,
    multiprocess_bidirection_hybrid_a_star_search,
    upsample_interpolation,
    downsample_smooth,
    downup_smooth,
    trajectory_add_zero_velocity,
)
from CarlaAutoParking.gridmap.gridmap import (
    GridMap,
    generate_gridmap_from_polygon,
    calc_likelihood_field,
)
from CarlaAutoParking.utils.parking_utils import (
    generate_obstacle_and_parking_polygon,
    generate_random_start_state,
    generate_goal_states_from_parking_rectpolygon,
    sort_goal_states,
    sort_parking_polygon,
)
from CarlaAutoParking.others.se2state import SE2State
from CarlaAutoParking.others.geometry import Polygon


def main():
    (
        obstacle_polygon_list,
        parking_polygon_list,
    ) = generate_obstacle_and_parking_polygon()

    TASK_NUM = 1
    """
    Task 0 (parallel parking). Task 1(T shape parking).
    """
    start_state_list = [
        SE2State(4, 15, -3.12),
        # SE2State(8.8, 16, -3.14 + 1.05),
        # SE2State(2, 12, 0.4),
        SE2State(15.90, 14.03, 0.38),
    ]

    start_state = start_state_list[TASK_NUM]

    goal_state_list = generate_goal_states_from_parking_rectpolygon(
        parking_polygon_list[TASK_NUM]
    )

    goal_state_list = sort_goal_states(goal_state_list, start_state)
    goal_state = goal_state_list[0]

    """
    Plot task.
    """
    plt.figure(0, figsize=[8, 8])
    plt_utils.plot_task(obstacle_polygon_list, start_state, goal_state)
    plt.draw()
    plt.pause(0.1)
    """
    Gridmap
    """

    grid_start = time.time()
    gridmap = generate_gridmap_from_polygon(
        obstacle_polygon_list, parking_polygon_list, start_state
    )

    print(f"create gridmap time is {time.time() - grid_start:10.5f}")
    """
    Search.
    """
    search_start = time.time()
    # path = hybrid_a_star_search(start_state, goal_state, gridmap)
    path = hybrid_a_star_reverse_search_with_analystic_expantion(
        goal_state, start_state, gridmap
    )
    # path = bidirection_hybrid_a_star_search(start_state, goal_state, gridmap)

    # path = multiprocess_bidirection_hybrid_a_star_search(
    #     start_state, goal_state, gridmap
    # )
    plt.figure(0, figsize=[8, 8])
    plt_utils.plot_path(path)

    print(f"search time is {time.time() - search_start:10.5f}")

    """
    Plot control.
    """
    plt.figure(1, [8, 12])
    plt_utils.plot_control(path)
    plt.show()
    # plt.pause(0.1)

    print("\033[32m==Finished frontend test.===\033[0m")

    file_name = "se2path" + str(TASK_NUM) + ".pickle"
    plt_utils.save_pickle_file(path, file_name)

    print("\033[32m==Save Search Result.===\033[0m")


def test_smooth():
    (
        obstacle_polygon_list,
        _,
    ) = generate_obstacle_and_parking_polygon()

    """
    Task 0 (parallel parking). Task 1(T shape parking).
    """

    TASK_NUM = 0
    file_name = "se2path" + str(TASK_NUM) + ".pickle"

    path = plt_utils.load_pickle_file(file_name)

    start_state = path[0]
    goal_state = path[-1]

    plt.figure(0, figsize=[8, 8])

    plt_utils.plot_task(obstacle_polygon_list, start_state, goal_state)
    plt_utils.plot_path(path, "path")

    interval = 3
    ds_path = downsample_smooth(path, interval)
    plt_utils.plot_path(ds_path, "ds_path")

    us_path = upsample_interpolation(ds_path, interval - 1)
    plt_utils.plot_path(us_path, "us_path")

    smooth_path = downup_smooth(path)
    plt_utils.plot_path(smooth_path, "smooth_path")

    zero_path = trajectory_add_zero_velocity(smooth_path)
    plt.draw()
    plt.pause(0.1)

    plt.figure(1, figsize=[8, 10])
    plt_utils.plot_control(path)
    # utils.plot_control(ds_path)
    # utils.plot_control(us_path)
    plt_utils.plot_control(smooth_path)
    plt_utils.plot_control(zero_path)

    plt.show()

    print("123")


def test_frontend():
    (
        obstacle_polygon_list,
        parking_polygon_list,
    ) = generate_obstacle_and_parking_polygon()

    TASK_NUM = 1
    """
    Task 0 (parallel parking). Task 1(T shape parking).
    """
    start_state_list = [
        SE2State(4, 15, 3.12),
        SE2State(8.8, 16, -3.14 + 1.05),
        # SE2State(2, 12, 0.4),
        SE2State(15.90, 14.03, 0.38),
    ]

    start_state = start_state_list[TASK_NUM]
    parking_polygon = sort_parking_polygon(parking_polygon_list, start_state)[0]
    goal_se2state = sort_goal_states(
        generate_goal_states_from_parking_rectpolygon(parking_polygon), start_state
    )[0]
    frontend = Frontend()
    frontend.initialize(obstacle_polygon_list, start_state, goal_se2state)

    """
    Plot task.
    """
    plt.figure(0, figsize=[8, 8])
    plt_utils.plot_task(obstacle_polygon_list, start_state, frontend.goal_se2state)
    plt_utils.plot_polygon(parking_polygon)
    plt.draw()
    plt.pause(0.1)

    frontend.search()
    search_path = frontend.extract_path()
    plt_utils.plot_path(search_path)
    plt.draw()
    plt.pause(0.1)

    plt.figure(1, figsize=[8, 10])
    plt_utils.plot_control(search_path)

    plt.show()

    file_name = "se2path" + str(TASK_NUM) + ".pickle"
    plt_utils.save_pickle_file(search_path, file_name)

    print("\033[32m==Save Result.===\033[0m")


if __name__ == "__main__":
    # main()
    # test_frontend()

    test_smooth()
