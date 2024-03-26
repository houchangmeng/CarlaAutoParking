import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from CarlaAutoParking.planner.search import breadth_first_search
from CarlaAutoParking.gridmap.gridmap import GridMap, calc_likelihood_field
import CarlaAutoParking.utils.plt_utils as plt_utils

from CarlaAutoParking.others.se2state import SE2State


def main():
    grid_map = GridMap()
    start = np.array([2, 2, 0])
    goal = np.array([10, 10, 0])
    start_se2 = SE2State(start[0], start[1], start[2])
    goal_se2 = SE2State(goal[0], goal[1], goal[2])

    heatmap = breadth_first_search(goal_se2, start_se2, grid_map)
    heatmap = np.array(heatmap) * 5

    heatmap = calc_likelihood_field(grid_map, goal_se2)
    plt_utils.plot_heatmap(heatmap)
    plt.draw()
    plt.show()


if __name__ == "__main__":
    main()
