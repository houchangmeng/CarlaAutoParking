import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import CarlaAutoParking.env.carla_utils as carla_utils
import CarlaAutoParking.utils.plt_utils as plt_utils
import numpy as np
import matplotlib.pyplot as plt
import carla
from CarlaAutoParking.controller.carla_controller import CarlaParkingController
from typing import List


def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    sim_world: carla.World = client.load_world("Town05_Opt")
    sim_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    dt = 0.02
    settings = sim_world.get_settings()
    settings.substepping = True
    settings.max_substep_delta_time = dt / 16
    settings.max_substeps = 16
    settings.fixed_delta_seconds = dt
    settings.synchronous_mode = True

    sim_world.apply_settings(settings)

    debug = sim_world.debug

    file_name = "carla_se2state_list_local_rear.pickle"
    local_rear_ref_traj = plt_utils.load_pickle_file(file_name)

    file_name = "carla_se2state_list_local_center.pickle"
    local_center_ref_traj = plt_utils.load_pickle_file(file_name)

    file_name = "carla_se2state_list_global_rear.pickle"
    global_rear_ref_traj = plt_utils.load_pickle_file(file_name)

    file_name = "carla_se2state_list_global_center.pickle"
    global_center_ref_traj: List = plt_utils.load_pickle_file(file_name)

    """
    Change rear or center.
    """
    reference_trajectory = global_center_ref_traj
    ego = carla_utils.spawn_parking_ego_car(sim_world)
    control = carla.VehicleControl()
    control.throttle = 0.1
    carla_utils.set_spectator(ego)
    ego.apply_control(control)

    for _ in range(100):
        ego.apply_control(control)
        sim_world.tick()

    chassis_controller = CarlaParkingController()
    chassis_controller.set_reference_trajectory(reference_trajectory)
    N = len(reference_trajectory)
    tt = np.linspace(0, N * dt, N)

    current_se2state = carla_utils.generate_center_se2state_from_vehicle(ego)
    current_se2state.t = 0.0

    tracking_trajectory = []
    throttle_list = []
    steer_list = []

    for i in range(N):
        sim_world.tick()
        # for _ in range(10):
        #     sim_world.tick()

        carla_utils.draw_path(debug, reference_trajectory, color=(255, 0, 0))
        carla_utils.draw_path(
            debug, tracking_trajectory, color=(0, 255, 0), thickness=0.2
        )
        carla_utils.set_spectator(ego)

        vehicle_center_se2state = carla_utils.generate_center_se2state_from_vehicle(ego)
        vehicle_rear_se2state = carla_utils.generate_rear_se2state_from_vehicle(ego)

        current_se2state = vehicle_center_se2state
        current_se2state.t = 0.02 * (i + 1)

        control = chassis_controller.action(current_se2state)
        ego.apply_control(control)

        tracking_trajectory += [current_se2state]

        throttle_list += [
            control.throttle if control.throttle > 0.01 else control.brake * (-1)
        ]
        steer_list += [control.steer]
        # time.sleep(0.02)

    carla_utils.draw_path(debug, reference_trajectory, color=(255, 0, 0))
    carla_utils.draw_path(debug, tracking_trajectory, color=(0, 255, 0))
    sim_world.tick()

    ego.destroy()

    # if recorder_time > 0:
    #     time.sleep(recorder_time)

    plt.figure(0, figsize=[8, 12])
    plt_utils.plot_control(reference_trajectory)
    plt_utils.plot_control(tracking_trajectory)
    plt.draw()
    plt.pause(0.1)

    throttle_array = np.array(throttle_list)
    steer_array = np.array(steer_list)
    plt.figure(1, figsize=[8, 12])
    plt.subplot(211)
    plt.plot(tt, throttle_array)
    plt.subplot(212)
    plt.plot(tt, steer_array)
    plt.show()

    print(client.stop_recorder())


import pygame
from pygame.locals import K_DOWN
from pygame.locals import K_LEFT
from pygame.locals import K_RIGHT
from pygame.locals import K_UP
from pygame.locals import K_q


"""
TODO

1. throttle table. # TODO
2. Keyboard control.
"""

if __name__ == "__main__":
    import time

    main()
    # time.sleep(5)
    # test_autopolit()
