import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from typing import TypeVar, List
import CarlaAutoParking.utils.plt_utils as plt_utils
import CarlaAutoParking.utils.parking_utils as st
from CarlaAutoParking.planner.search import Frontend, trajectory_add_zero_velocity
from CarlaAutoParking.others.se2state import SE2State, generate_vehicle_vertexes
import sys


from CarlaAutoParking.others.coord_transform import (
    move_polygon_list,
    move_se2state_list,
)

Vehicle = TypeVar("Vehicle")


class State:
    def handle(self, vehicle: Vehicle):
        pass

    def __repr__(self) -> str:
        return f"State baseclass"


class ParkingFail(State):
    def handle(self, vehicle: Vehicle):
        print(f"Current state is : {vehicle.state}")
        print("\033[31m=== Parking Failed. ===\033[0m")
        sys.exit(1)

    def __repr__(self) -> str:
        return f"ParkingFail"


class ParkingSucess(State):
    def handle(self, vehicle: Vehicle):
        # vehicle.step()
        print(f"Current state is : {vehicle.state}")
        print("\033[32m=== Parking Success. ===\033[0m")

        # plt.figure(10, figsize=[8, 10])
        # plt_utils.plot_control(vehicle.local_reference_trajectory)
        # plt_utils.plot_control(vehicle.local_tracking_trajectory)
        # plt.show()

        vehicle.env.close()
        vehicle.gui.close()

    def __repr__(self) -> str:
        return f"ParkingSucess"


class Perception(State):
    def handle(self, vehicle: Vehicle):
        if (
            vehicle.current_center_se2state is not None
            and len(vehicle.global_parking_polygons) > 0
        ):
            vehicle.set_state(Decision())

        print(f"Current state is : {vehicle.state}")

    def __repr__(self) -> str:
        return f"Perception"


import CarlaAutoParking.env.carla_utils as carla_utils

import random

from CarlaAutoParking.others.geometry import point_in_polygon, polygon_in_polygon


class Decision(State):
    def handle(self, vehicle: Vehicle):
        if vehicle.gui.select_point is not None:
            goal_point = vehicle.gui.select_point
            goal_parking_polygon = None
            for parking_polygon in vehicle.global_parking_polygons:
                if point_in_polygon(goal_point, parking_polygon):
                    goal_parking_polygon = parking_polygon
                    break

            if goal_parking_polygon is None:
                vehicle.gui.select_point = None
                print("\033[33m=== Select a reasonable point. ===\033[0m")
                time.sleep(1.0)
                return

            goal_rear_se2statelist = st.generate_goal_states_from_parking_rectpolygon(
                goal_parking_polygon
            )

            goal_rear_se2state = st.sort_selected_states(
                goal_rear_se2statelist, goal_point
            )[0]

            vehicle.gui.update_vehicle(goal_rear_se2state, "-m")
            goal_center_se2state = carla_utils.get_center_se2state_from_rear(
                goal_rear_se2state
            )

            vehicle.goal_center_se2state = goal_center_se2state
            vehicle.goal_rear_se2state = goal_rear_se2state

            vehicle.set_state(Planing())

        else:
            pass

        print(f"Current state is : {vehicle.state}")

    def __repr__(self) -> str:
        return f"Decision"


from CarlaAutoParking.planner.optimizer import Optimizer


class Planing(State):
    def __init__(self) -> None:
        super().__init__()
        self.planning_count = 0

    def handle(self, vehicle: Vehicle):
        """
        frontend search. if failed . set planning.
        backend optimization. if failed. set planning.
        count +=1
        if successed, set tracking
        """

        # try:

        print("\033[32m===Start frontend search.===\033[0m")

        frontend = Frontend()
        frontend.initialize(
            vehicle.global_obstacle_polygons,
            vehicle.current_rear_se2state,
            vehicle.goal_rear_se2state,
        )
        frontend.search()
        frontend_path = frontend.extract_path()
        vehicle.gui.update_path(frontend_path, "frontend")

        print("\033[32m===Start backend Optimize.===\033[0m")

        opti = Optimizer()
        opti.initialize(
            frontend_path, vehicle.global_obstacle_polygons, frontend.grid_map
        )
        opti.solve()

        local_rear_ref_traj_no_zero = opti.extract_result(
            current_time=vehicle.current_center_se2state.t
        )

        local_rear_ref_traj = trajectory_add_zero_velocity(
            local_rear_ref_traj_no_zero, 10
        )

        vehicle.gui.update_path(local_rear_ref_traj, "backend")

        local_center_ref_traj = (
            carla_utils.generate_center_se2state_list_from_rear_list(
                local_rear_ref_traj
            )
        )

        # """
        # save trajectory.
        # """

        global_rear_ref_traj = move_se2state_list(
            local_rear_ref_traj, vehicle.env.start_global_se2
        )

        global_center_ref_traj = move_se2state_list(
            local_center_ref_traj, vehicle.env.start_global_se2
        )

        # file_name = "carla_se2state_list_local_rear.pickle"
        # plt_utils.save_pickle_file(local_rear_ref_traj, file_name)

        # file_name = "carla_se2state_list_local_center.pickle"
        # plt_utils.save_pickle_file(local_center_ref_traj, file_name)

        # file_name = "carla_se2state_list_global_rear.pickle"
        # plt_utils.save_pickle_file(global_rear_ref_traj, file_name)

        # file_name = "carla_se2state_list_global_center.pickle"
        # plt_utils.save_pickle_file(global_center_ref_traj, file_name)

        """
        Load trajectory.
        """

        # file_name = "carla_se2state_list_local_rear.pickle"
        # local_rear_ref_traj = plt_utils.load_pickle_file(file_name)

        # file_name = "carla_se2state_list_local_center.pickle"
        # local_center_ref_traj = plt_utils.load_pickle_file(file_name)

        # file_name = "carla_se2state_list_global_rear.pickle"
        # global_rear_ref_traj = plt_utils.load_pickle_file(file_name)

        # file_name = "carla_se2state_list_global_center.pickle"
        # global_center_ref_traj: List = plt_utils.load_pickle_file(file_name)

        vehicle.env.set_tracking_task(global_center_ref_traj)
        vehicle.local_reference_trajectory = local_rear_ref_traj
        vehicle.local_tracking_trajectory = []

        print(f"Current state is : {vehicle.state}")

        vehicle.controller.set_reference_trajectory(local_rear_ref_traj)
        vehicle.set_state(Tracking())

        # except Exception as e:
        # print(f"\033[31m Catch Exception {e} \033[0m")
        # print(f"Replan ...{self.planning_count:4d}")
        # self.planning_count += 1

        """
        1. TODO reduce discrete grid size if a star queue is empty.
        """

        if self.planning_count >= 5:
            vehicle.set_state(ParkingFail())

    def __repr__(self) -> str:
        return f"Planing"


class CheckCollision(State):
    def handle(self, vehicle: Vehicle):
        if vehicle.polygon_manager.check_collision(
            vehicle.current_rear_se2state, vehicle.local_reference_trajectory
        ):
            u = vehicle.controller.emergency_stop()
            print(
                f"\033[33m curre trajectory has collision with some obstacle. replaning.\033[0m"
            )
            vehicle.gui.select_point = None
            vehicle.set_state(Decision())
        else:
            u = vehicle.controller.action(vehicle.current_rear_se2state)
            vehicle.set_state(Tracking())

        vehicle.step(u)

        print(f"Current state is : {vehicle.state}, time is {time.time():10.2f}")

    def __repr__(self) -> str:
        return f"CheckCollision"


class Tracking(State):
    def handle(self, vehicle: Vehicle):
        vehicle.local_tracking_trajectory += [vehicle.current_rear_se2state]
        vehicle.env_action = vehicle.controller.action(vehicle.current_rear_se2state)

        """
        Check collision.
        """

        if (
            int(vehicle.env.step_count)
            % vehicle.polygon_manager.collision_check_interval
            == 0.1
        ):
            vehicle.set_state(CheckCollision())

        print(f"Current state is : {vehicle.state}, time is {time.time():10.2f}")

    def __repr__(self) -> str:
        return f"Tracking"


from CarlaAutoParking.others.geometry import (
    PolygonContainer,
    Polygon,
    polygon_intersect_polygon,
)


class PolygonManager:
    def __init__(self) -> None:
        self.global_obstacle_polygon_container: PolygonContainer = PolygonContainer()
        self.global_parking_polygon_container: PolygonContainer = PolygonContainer()
        self.collision_check_interval = 100

    def check_collision(
        self, current_se2state: SE2State, reference_trajectory: List[SE2State]
    ):
        index = 0

        for se2state in reference_trajectory:
            if current_se2state.t > se2state.t:
                index += 1

        for i in range(index, len(reference_trajectory)):
            se2state = reference_trajectory[i]
            vehicle_vertices = generate_vehicle_vertexes(se2state)
            vehicle_polygon = Polygon(vehicle_vertices)
            for obstacle_polygon in self.global_obstacle_polygon_container:
                if polygon_intersect_polygon(vehicle_polygon, obstacle_polygon):
                    print(f"\033[31m==Collision happened.==\033[0m")
                    print(f"CheckCollision se2 state is...{se2state}")
                    return True

        return False


Point = TypeVar("Point")
import carla

from env.parking_carla_env import CarlaEnv
from CarlaAutoParking.controller.carla_controller import CarlaParkingController


class Vehicle:
    def __init__(self) -> None:
        self.state: State = None
        self.polygon_manager: PolygonManager = None

        self.env: CarlaEnv = None
        self.path_opti: Optimizer = None
        self.controller: CarlaParkingController = None
        self.file_name: str = None
        self.frontend: Frontend = None

        self.action_count = None

        self.local_reference_trajectory = None
        self.local_tracking_trajectory = None
        self.current_center_se2state = None
        self.current_rear_se2state = None
        self.goal_center_se2state = None
        self.goal_rear_se2state = None
        self.env_action = None

        self.gui = plt_utils.GUI()

    @property
    def global_obstacle_polygons(self):
        return self.polygon_manager.global_obstacle_polygon_container

    @property
    def global_parking_polygons(self):
        return self.polygon_manager.global_parking_polygon_container

    def initialize(self):
        self.env = CarlaEnv()
        self.env.reset()
        self.polygon_manager = PolygonManager()
        self.path_opti = Optimizer()
        self.action_count = 0
        self.controller = CarlaParkingController()
        self.local_tracking_trajectory = []
        self.local_reference_trajectory = []
        self.current_center_se2state = None
        self.current_rear_se2state = None
        self.goal_center_se2state = None
        self.goal_rear_se2state = None
        self.env_action = carla.VehicleControl()

        self.set_state(Perception())

    def set_state(self, state: State):
        self.state = state

    def add_global_obstacle_polygon(self, obstacle_polygon_list: List[Polygon]):
        if len(obstacle_polygon_list) == 0:
            return
        for obstacle_polygon in obstacle_polygon_list:
            self.polygon_manager.global_obstacle_polygon_container.__iadd__(
                obstacle_polygon
            )

    def add_global_parking_polygon(self, parking_polygon_list: List[Polygon]):
        if len(parking_polygon_list) == 0:
            return

        for parking_polygon in parking_polygon_list:
            self.polygon_manager.global_parking_polygon_container += parking_polygon

        for obstacle_polygon in self.polygon_manager.global_obstacle_polygon_container:
            for (
                parking_polygon
            ) in self.polygon_manager.global_parking_polygon_container:
                if polygon_intersect_polygon(obstacle_polygon, parking_polygon):
                    self.polygon_manager.global_parking_polygon_container -= (
                        parking_polygon
                    )

    def action(self):
        current_se2state, _, _, done, info = self.env.step(self.env_action)

        global_obstacle_polygon = move_polygon_list(
            info["local_obstacle_polygon_list"], current_se2state.se2
        )

        global_parking_polygon = move_polygon_list(
            info["local_parking_polygon_list"], current_se2state.se2
        )

        # self.add_global_obstacle_polygon(info["global_obstacle_polygon_list"])
        # self.add_global_parking_polygon(info["global_parking_polygon_list"])

        self.add_global_obstacle_polygon(global_obstacle_polygon)
        self.add_global_parking_polygon(global_parking_polygon)
        self.current_center_se2state = current_se2state
        self.current_rear_se2state = carla_utils.get_rear_se2state_from_center(
            current_se2state
        )

        self.update_gui()

        if done:
            for _ in range(10):
                u = self.controller.emergency_stop()
                self.env.step(u)
            self.set_state(ParkingSucess())

        self.state.handle(self)

    def update_gui(self):
        if type(self.state).__name__ == "Tracking":
            if self.env.step_count % 50 == 0:
                self.gui.update_vehicle(self.current_rear_se2state)
                # self.gui.update_polygon(self.global_obstacle_polygons)
                # self.gui.update_polygon(self.global_parking_polygons, line_type="-g")
        else:
            self.gui.update_vehicle(self.current_rear_se2state)
            self.gui.update_polygon(self.global_obstacle_polygons)
            self.gui.update_polygon(self.global_parking_polygons, line_type="-g")


"""
TODO:

1. If some parking polygon has been occupied. this parking polygon should be remove. # OK
"""

import time
import random


def main():
    print("\033[32m=== Simulation Start. ===\033[0m")

    veh = Vehicle()
    veh.initialize()
    count = 0

    while True:
        veh.action()
        count += 1
        if isinstance(veh.state, ParkingSucess):
            sys.exit(0)


if __name__ == "__main__":
    main()
