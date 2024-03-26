import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from CarlaAutoParking.utils.parking_utils import generate_obstacle_and_parking_polygon
import matplotlib.pyplot as plt
import pickle
import CarlaAutoParking.utils.plt_utils as plt_utils
from CarlaAutoParking.planner.serach_utils import (
    upsample_interpolation,
    downsample_smooth,
    downup_smooth,
)
from CarlaAutoParking.planner.optimizer import Optimizer
from CarlaAutoParking.gridmap.gridmap import GridMap, generate_gridmap_from_polygon

TASK_NUM = 0


def test_optimizer():
    (
        obstacle_polygon_list,
        parking_polygon_list,
    ) = generate_obstacle_and_parking_polygon()

    file_name = "se2path" + str(TASK_NUM) + ".pickle"

    search_path = plt_utils.load_pickle_file(file_name)

    start_se2state = search_path[0]
    goal_se2state = search_path[-1]

    plt.figure(0, figsize=[8, 8])

    plt_utils.plot_task(obstacle_polygon_list, start_se2state, goal_se2state)
    plt_utils.plot_path(search_path, "search_path")
    plt.draw()
    plt.pause(0.1)

    # dt = 0.02
    # interval = int((search_path[1].t - search_path[0].t) / dt) - 1
    # us_path = upsample_interpolation(search_path, interval)
    # us_path = downup_smooth(us_path)

    interval = 10
    us_path = search_path

    plt_utils.plot_path(us_path, "us_path")
    plt.draw()
    plt.pause(0.1)

    gridmap = generate_gridmap_from_polygon(
        obstacle_polygon_list, parking_polygon_list, start_se2state
    )

    opti = Optimizer()
    opti.initialize(us_path, obstacle_polygon_list, gridmap)
    opti.solve()

    opti_path = opti.extract_result()
    plt_utils.plot_path(opti_path, "opti_path")
    plt.draw()
    plt.pause(0.1)
    # ds_opti_path = downsample_smooth(opti_path, interval)
    # plt_utils.plot_trajectory_animation(ds_opti_path)
    plt.draw()
    plt.pause(0.1)

    plt.figure(1, figsize=[8, 8])
    plt_utils.plot_control(search_path)
    plt_utils.plot_control(us_path)
    plt_utils.plot_control(opti_path)

    plt.show()

    file_name = "se2opti_path" + str(TASK_NUM) + ".pickle"
    plt_utils.save_pickle_file(opti_path, file_name)

    print("\033[32m==Save Optimizer Result.===\033[0m")


if __name__ == "__main__":
    test_optimizer()
