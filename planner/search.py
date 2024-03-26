import numpy as np
from typing import List, Tuple
from queue import PriorityQueue, Queue
from CarlaAutoParking.others.se2state import SE2State
from config import VehicleConfig, SearchConfig
from CarlaAutoParking.gridmap.gridmap import (
    GridMap,
    generate_gridmap_from_parking_task,
)
from CarlaAutoParking.others.geometry import Polygon


class Frontend:
    def __init__(self) -> None:
        self.goal_se2state: SE2State = None
        self.start_se2state: SE2State = None
        self.grid = None
        self.veh_cfg = VehicleConfig()
        self.search_cfg = SearchConfig()
        self.path = []

    def initialize(
        self,
        obstacle_polygon_list: List[Polygon],
        start_se2state: SE2State,
        goal_se2state: SE2State,
    ):
        self.grid_map = generate_gridmap_from_parking_task(
            obstacle_polygon_list, start_se2state, goal_se2state
        )
        self.start_se2state = start_se2state
        self.goal_se2state = goal_se2state

    def search(self):
        self.path = hybrid_a_star_reverse_search_with_analystic_expantion(
            self.start_se2state,
            self.goal_se2state,
            self.grid_map,
            self.veh_cfg,
            self.search_cfg,
        )

    def extract_path(self, dt=0.02):
        if len(self.path) < 5:
            raise ValueError("Path too short.")

        interval = int((self.path[1].t - self.path[0].t) / dt) - 1
        if interval > 3:
            return upsample_interpolation(self.path, interval)
        else:
            raise ValueError("dt too big.")


from CarlaAutoParking.planner.serach_utils import *

from CarlaAutoParking.gridmap.gridmap import (
    GridMap,
    outboundary,
    collision,
    generate_heuristic_map,
    generate_obstacle_grid,
    generate_visited_map_2d,
    generate_visited_map_3d,
)


def breadth_first_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
):
    q = Queue()
    close_list: list[SE2State] = []
    open_dict: dict[Tuple, SE2State] = dict()

    start.cost_to_here = start.cost_to_gridstate(start)
    start.cost_to_goal = start.cost_to_gridstate(goal)
    q.put(start)

    open_dict[start.get_index_2d()] = start

    print("Heuristic initializing")
    visited_map = generate_visited_map_2d(grid_map)
    obstacle_map = generate_obstacle_grid(grid_map)
    heuristic = generate_heuristic_map(grid_map)

    print("Heuristic start search")

    it = 0
    while not q.empty():
        current_state: SE2State = q.get()
        it += 1
        if it % 10000 == 0:
            print(f"Heuristic search iter {it:5d}")

        visited_map[current_state.x_index][current_state.y_index] = True
        heuristic[current_state.x_index][
            current_state.y_index
        ] = current_state.cost_to_here

        close_list += [current_state]

        for next_state in get_next_grid_state(current_state, grid_map):
            if outboundary(grid_map, next_state):
                continue
            elif obstacle_map[next_state.x_index][next_state.y_index]:
                continue
            elif visited_map[next_state.x_index][next_state.y_index]:
                continue
            else:
                next_state.cost_to_here = (
                    current_state.cost_to_here
                    + current_state.cost_to_gridstate(next_state)
                )

                next_index = next_state.get_index_2d()
                if next_index in open_dict:
                    if open_dict[next_index].cost_to_here > next_state.cost_to_here:
                        open_dict[next_index].cost_to_here = next_state.cost_to_here
                        open_dict[next_index].parent = current_state
                else:
                    open_dict[next_state.get_index_2d()] = next_state
                    q.put(next_state)

    print(f"Heuristic search finished, iter {it:5d} ")

    return heuristic


def hybrid_a_star_search_step(
    q: PriorityQueue,
    open_dict: dict,
    close_list: list,
    heuristic_weight: float,
    penalty_change_gear: int,
    grid_map: GridMap,
    heuristic_map: List[List[float]],
    visited_map: List[List[bool]],
    veh_cfg=VehicleConfig(),
    search_cfg=SearchConfig(),
):
    """
    all parameter is passed by reference.
    """
    if q.empty():
        return

    current_state: SE2State = q.get()[1]

    current_state.visited = True

    visited_map[current_state.x_index][current_state.y_index][
        current_state.heading_index
    ] = True

    close_list += [current_state]

    for next_state in get_next_states(current_state, grid_map, veh_cfg, search_cfg):
        if collision(grid_map, next_state):
            continue
        elif visited_map[next_state.x_index][next_state.y_index][
            next_state.heading_index
        ]:
            continue
        else:
            cost_to_here = current_state.cost_to_here + current_state.cost_to_state(
                next_state
            )
            if (
                next_state.direction_index > 3 and current_state.direction_index <= 3
            ) or (
                next_state.direction_index <= 3 and current_state.direction_index > 3
            ):
                """Different direction."""
                cost_to_here = cost_to_here * penalty_change_gear

            else:
                """Same direction."""
                pass

            next_state.cost_to_here = cost_to_here
            next_index = next_state.get_index_3d()
            next_state.parent = current_state

            next_state.cost_to_goal = (
                heuristic_weight * heuristic_map[next_state.x_index][next_state.y_index]
            )

            try:
                if next_index in list(open_dict.keys()):
                    if open_dict[next_index].cost() > next_state.cost():
                        open_dict[next_index].cost_to_goal = next_state.cost_to_goal
                        open_dict[next_index].cost_to_here = next_state.cost_to_here
                        open_dict[next_index].parent = current_state
                else:
                    open_dict[next_state.get_index_3d()] = next_state
                    q.put((next_state.cost(), next_state), block=False)
            except:
                print("Queue has full")


def hybrid_a_star_reverse_search_with_analystic_expantion(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    veh_cfg=VehicleConfig(),
    search_cfg=SearchConfig(),
):
    import faulthandler

    faulthandler.enable()

    compute_3d_index(grid_map, start)
    compute_3d_index(grid_map, goal)

    print(f"Start : {start}")
    print(f"Goal : {goal}")

    temp_goal = goal
    goal = deepcopy(start)
    start = deepcopy(temp_goal)

    max_it = search_cfg.max_iteration
    penalty_change_gear = search_cfg.penalty_change_gear
    heuristic_weight = search_cfg.heuristic_weight

    visited_map = generate_visited_map_3d(grid_map)

    heuristic_map = breadth_first_search(goal, start, grid_map)

    start.cost_to_here = start.cost_to_state(start)
    start.cost_to_goal = heuristic_map[start.x_index][start.y_index]

    q = PriorityQueue()
    close_list = []

    open_dict = dict()
    open_dict[start.get_index_3d()] = start
    q.put((start.cost(), start))

    goal_set = get_next_states(goal, grid_map, veh_cfg, search_cfg)

    it = -1

    print("Hybrid a star start seaching")

    while ((not q.empty())) and (it < max_it):
        it += 1
        if it % (10 * search_cfg.analystic_expantion_interval) == 0:
            print(f"Hybrid seaching , iter : {it:5d}")

        hybrid_a_star_search_step(
            q,
            open_dict,
            close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            heuristic_map,
            visited_map,
            veh_cfg,
            search_cfg,
        )

        current_state = close_list[-1]

        if it % search_cfg.analystic_expantion_interval == 0:
            success, path2 = update_path_with_analystic_expantion(
                current_state, goal, grid_map
            )

            if success:
                print(f"Find goal. Iteration {it:5d}")
                path1 = back_track_state(current_state)
                path = path1 + path2

                path = path[::-1]

                for i in range(0, len(path)):
                    path[i].v *= -1
                    path[i].t = i * veh_cfg.T

                return path

        if current_state in goal_set:
            print(f"Find goal. Iteration {it:5d}")
            path = back_track_close(close_list)
            path = path[::-1]

            for i in range(0, len(path)):
                path[i].v *= -1
                path[i].t = i * veh_cfg.T

            return path

    print(f"Search failed, Iteration {it:5d}")
    return back_track_close(close_list)


def hybrid_a_star_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    import faulthandler

    faulthandler.enable()

    compute_3d_index(grid_map, start)
    compute_3d_index(grid_map, goal)

    print(f"Start : {start}")
    print(f"Goal : {goal}")

    max_it = search_cfg.max_iteration
    max_heading_index_error = search_cfg.max_heading_index_error
    penalty_change_gear = search_cfg.penalty_change_gear
    heuristic_weight = search_cfg.heuristic_weight

    visited_map = generate_visited_map_3d(grid_map)

    heuristic_map1 = breadth_first_search(goal, start, grid_map)

    # import matplotlib.pyplot as plt
    # import utils

    # heuristic_map2 = calc_likelihood_field(grid_map, goal)
    # plt.figure(10, figsize=[8, 8])
    # utils.plot_heatmap(heuristic_map2)
    # plt.pause(0.2)

    # heuristic_map = 0.5 * (np.array(heuristic_map1) + np.array(heuristic_map2))

    heuristic_map = heuristic_map1

    start.cost_to_here = start.cost_to_state(start)
    start.cost_to_goal = heuristic_map[start.x_index][start.y_index]
    q = PriorityQueue()
    close_list = []

    open_dict = dict()
    open_dict[start.get_index_3d()] = start
    q.put((start.cost(), start))

    goal_set = get_next_states(goal, grid_map)

    it = 0

    print("Hybrid a star start seaching")

    while ((not q.empty())) and (it < max_it):
        it += 1
        if it % 10000 == 0:
            print(f"Hybrid seaching , iter : {it:5d}")

        hybrid_a_star_search_step(
            q,
            open_dict,
            close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            heuristic_map,
            visited_map,
        )
        current_state = close_list[-1]
        # analystic_state_list = analystic_expand(current_state, goal, grid_map)
        # if len(analystic_state_list) > 0:
        #     print(f"Find goal. Iteration {it:5d}")
        #     return back_track_close(close_list) + analystic_state_list

        if current_state in goal_set:
            # current_state.parent = goal
            print(f"Find goal. Iteration {it:5d}")
            return back_track_close(close_list)

    print(f"Search failed, Iteration {it:5d}")
    return back_track_close(close_list)


def hybrid_a_star_reverse_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    import faulthandler

    faulthandler.enable()

    compute_3d_index(grid_map, start)
    compute_3d_index(grid_map, goal)

    print(f"Start : {start}")
    print(f"Goal : {goal}")

    temp_goal = goal
    goal = deepcopy(start)
    start = deepcopy(temp_goal)

    max_it = search_cfg.max_iteration
    max_heading_index_error = search_cfg.max_heading_index_error
    penalty_change_gear = search_cfg.penalty_change_gear
    heuristic_weight = search_cfg.heuristic_weight

    visited_map = generate_visited_map_3d(grid_map)

    heuristic_map = breadth_first_search(goal, start, grid_map)

    start.cost_to_here = start.cost_to_state(start)
    start.cost_to_goal = heuristic_map[start.x_index][start.y_index]
    q = PriorityQueue()
    close_list = []

    open_dict = dict()
    open_dict[start.get_index_3d()] = start
    q.put((start.cost(), start))

    goal_set = get_next_states(goal, grid_map)

    it = 0

    print("Hybrid a star start seaching")

    while ((not q.empty())) and (it < max_it):
        it += 1
        if it % 10000 == 0:
            print(f"Hybrid seaching , iter : {it:5d}")

        hybrid_a_star_search_step(
            q,
            open_dict,
            close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            heuristic_map,
            visited_map,
        )
        current_state = close_list[-1]

        if current_state in goal_set:
            # current_state.parent = goal
            print(f"Find goal. Iteration {it:5d}")
            path = back_track_close(close_list)
            path = path[::-1]

            dt = abs(path[-1].t - path[-2].t)
            for i in range(0, len(path)):
                path[i].v *= -1
                path[i].t = i * dt

            return path

    print(f"Search failed, Iteration {it:5d}")
    return back_track_close(close_list)


def bidirection_hybrid_a_star_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    import faulthandler
    import matplotlib.pyplot as plt

    faulthandler.enable()

    compute_3d_index(grid_map, start)
    compute_3d_index(grid_map, goal)

    print(f"Start : {start}")
    print(f"Goal : {goal}")

    """
    Search Configure.
    """

    max_it = search_cfg.max_iteration
    heuristic_weight = search_cfg.heuristic_weight
    penalty_change_gear = search_cfg.penalty_change_gear
    visited_exam_interval = search_cfg.visited_exam_interval
    """
    Primitive init.
    """
    print("Primitive initializing")

    heuristic_map = breadth_first_search(goal, start, grid_map)

    # import utils
    # plt.figure(1, figsize=[8, 8])
    # heatmap = np.array(heuristic_map) * 5
    # utils.plot_heatmap(heatmap)
    # plt.draw()
    # plt.show()

    start.cost_to_here = start.cost_to_state(start)
    start.cost_to_goal = heuristic_map[start.x_index][start.y_index]
    q = PriorityQueue()
    q.put((start.cost(), start))

    open_dict: dict[Tuple, SE2State] = dict()
    open_dict[start.get_index_3d()] = start
    close_list = []
    visited_map = generate_visited_map_3d(grid_map)

    """
    Dual init.
    """

    print("Dual initializing")
    dual_start = deepcopy(goal)
    dual_goal = deepcopy(start)

    dual_visited_map = deepcopy(visited_map)

    dual_heuristic_map = breadth_first_search(dual_goal, dual_start, grid_map)

    # import utils

    # heatmap = np.array(dual_heuristic_map) * 5
    # utils.plot_heatmap(heatmap)
    # plt.draw()
    # plt.show()

    dual_start.cost_to_here = dual_start.cost_to_state(dual_start)
    dual_start.cost_to_goal = dual_heuristic_map[dual_start.x_index][dual_start.y_index]
    dual_q = PriorityQueue()
    dual_q.put((dual_start.cost(), dual_start))

    dual_open_dict: dict[Tuple, SE2State] = dict()
    dual_open_dict[dual_start.get_index_3d()] = dual_start
    dual_close_list = []

    it = 0

    print("Bidirection hybrid a star start seaching.")

    while True:
        it += 1

        if it > max_it:
            raise TimeoutError("Search has over max iteration.")

        if q.empty() and dual_q.empty():
            raise BufferError("All Queue is empty, Search failed.")

        hybrid_a_star_search_step(
            q,
            open_dict,
            close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            heuristic_map,
            visited_map,
        )
        hybrid_a_star_search_step(
            dual_q,
            dual_open_dict,
            dual_close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            dual_heuristic_map,
            dual_visited_map,
        )

        if it % visited_exam_interval == 0:
            merge_prim_index, merge_dual_index = visitedmap_check(
                visited_map, dual_visited_map, open_dict, dual_open_dict, grid_map
            )
            if merge_prim_index is None:
                print(f"Merge. Iteration {it:5d}, can not find goal.")
                continue

            print(f"Find goal. Iteration {it:5d}")
            return back_track_merge(
                open_dict[merge_prim_index], dual_open_dict[merge_dual_index]
            )


def visitedmap_check(
    visited_map: List[List[List[bool]]],
    dual_visited_map: List[List[List[bool]]],
    open_dict: dict,
    dual_open_dict: dict,
    grid_map: GridMap,
    search_cfg: SearchConfig = SearchConfig(),
):
    visited_array_3d = np.array(visited_map) * 1
    dual_visited_array_3d = np.array(dual_visited_map) * 1

    visited_array_2d = np.sum(visited_array_3d, -1)
    visited_array_2d = np.where(visited_array_2d > 0, 1, 0)
    dual_visited_array_2d = np.sum(dual_visited_array_3d, -1)
    dual_visited_array_2d = np.where(dual_visited_array_2d > 0, 1, 0)

    merge_2dmap = visited_array_2d + dual_visited_array_2d

    merge_prim_index = None
    merge_dual_index = None

    if np.any(merge_2dmap == 2):
        merge_2dindex = np.where(merge_2dmap == 2)
        index2d_array = np.array(merge_2dindex)
        _, num = index2d_array.shape

        min_cost = np.inf

        for i in range(num):
            visited_heading_index = np.where(
                visited_array_3d[index2d_array[0, i], index2d_array[1, i]] == 1
            )
            visited_heading_index = np.sort(visited_heading_index).flatten()

            dual_visited_heading_index = np.where(
                dual_visited_array_3d[index2d_array[0, i], index2d_array[1, i]] == 1
            )
            dual_visited_heading_index = np.sort(dual_visited_heading_index).flatten()

            if len(visited_heading_index) < 1 or len(dual_visited_heading_index) < 1:
                continue

            prim_heading_index = None
            min_heading_index_error = int(1e8)

            for pi in visited_heading_index:
                for di in dual_visited_heading_index:
                    heading_index_error = abs(pi - di)
                    if heading_index_error <= search_cfg.max_heading_index_error:
                        # if (heading_index_error <= search_cfg.max_heading_index_error) or (
                        #     abs(heading_index_error - grid_map.headings * 0.5)
                        #     <= search_cfg.max_heading_index_error
                        # ):
                        if heading_index_error < min_heading_index_error:
                            min_heading_index_error = heading_index_error
                            prim_heading_index = pi
                            dual_heading_index = di

            if prim_heading_index is None:
                continue

            prim_index3d = (
                index2d_array[0, i],
                index2d_array[1, i],
                prim_heading_index,
            )

            dual_index3d = (
                index2d_array[0, i],
                index2d_array[1, i],
                dual_heading_index,
            )

            cost = (
                open_dict[prim_index3d].cost_to_here
                + dual_open_dict[dual_index3d].cost_to_here
                + min_heading_index_error
            )

            if cost < min_cost:
                min_cost = cost
                merge_prim_index = prim_index3d
                merge_dual_index = dual_index3d

    return merge_prim_index, merge_dual_index


import multiprocessing


def hybrid_a_star_worker(
    stop_event: multiprocessing.Event,
    visited_map_q: multiprocessing.Queue,
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    """
    worker
    """
    import faulthandler
    import os

    faulthandler.enable()

    max_it = search_cfg.max_iteration
    penalty_change_gear = search_cfg.penalty_change_gear
    heuristic_weight = search_cfg.heuristic_weight
    merge_interval = search_cfg.visited_exam_interval
    visited_map = generate_visited_map_3d(grid_map)

    heuristic_map = breadth_first_search(goal, start, grid_map)
    start.cost_to_here = start.cost_to_state(start)
    start.cost_to_goal = heuristic_map[start.x_index][start.y_index]
    q = PriorityQueue()
    close_list = []

    open_dict = dict()
    open_dict[start.get_index_3d()] = start
    q.put((start.cost(), start))

    it = 0

    print(f"PID {os.getpid()} start searching")

    while it < max_it:
        it += 1

        if stop_event.is_set():
            print(f"PID {os.getpid()} find goal.")
            break

        if q.empty() and it % merge_interval == 0:
            print(f"PID {os.getpid()} empty.")

        if visited_map_q.empty() and it % merge_interval == 0:
            print(f"PID {os.getpid()} search, iter {it:5d}")

            visited_map_q
            dict_pass = deepcopy(open_dict)
            visited_pass = deepcopy(visited_map)
            visited_map_q.put((visited_pass, dict_pass))

        hybrid_a_star_search_step(
            q,
            open_dict,
            close_list,
            heuristic_weight,
            penalty_change_gear,
            grid_map,
            heuristic_map,
            visited_map,
        )

    while not visited_map_q.empty():
        visited_map_q.get()  # clear the queue
    visited_map_q.close()


def multiprocess_bidirection_hybrid_a_star_search(
    start: SE2State,
    goal: SE2State,
    grid_map: GridMap,
    search_cfg=SearchConfig(),
):
    import faulthandler
    import time

    faulthandler.enable()

    """
    Search Configure.
    """
    print(f"Start : {start}")
    print(f"Goal : {goal}")

    compute_3d_index(grid_map, start)
    compute_3d_index(grid_map, goal)
    """
    Primitive.
    """
    stop_event = multiprocessing.Event()

    q1 = multiprocessing.Queue(maxsize=2)
    worker1 = multiprocessing.Process(
        target=hybrid_a_star_worker,
        args=(stop_event, q1, start, goal, grid_map, search_cfg),
    )
    worker1.start()

    """
    Dual.
    """

    dual_start = deepcopy(goal)
    dual_goal = deepcopy(start)

    q2 = multiprocessing.Queue(maxsize=2)
    worker2 = multiprocessing.Process(
        target=hybrid_a_star_worker,
        args=(stop_event, q2, dual_start, dual_goal, grid_map, search_cfg),
    )
    worker2.start()

    # print("Multiprocessing bidirection hybrid a star start.")

    maxit = search_cfg.max_iteration / search_cfg.visited_exam_interval

    it = 0
    visited_map, dual_visited_map = None, None

    while True:
        it += 1
        if it > maxit:
            raise TimeoutError("Bidirection Hybrid A Star Search Failed.")

        time.sleep(0.1)

        if not q1.empty():
            visited_map, open_dict = q1.get()

        if not q2.empty():
            dual_visited_map, dual_open_dict = q2.get()

        if visited_map is not None and dual_visited_map is not None:
            merge_prim_index, merge_dual_index = visitedmap_check(
                visited_map, dual_visited_map, open_dict, dual_open_dict
            )
            if merge_prim_index is not None and merge_dual_index is not None:
                break

    q1.close()
    q2.close()

    worker1.terminate()
    worker2.terminate()

    # print("\033[32m=== Search Success. ===\033[0m")

    return back_track_merge(
        open_dict[merge_prim_index], dual_open_dict[merge_dual_index]
    )
