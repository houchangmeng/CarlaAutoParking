import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import CarlaAutoParking.env.carla_utils as carla_utils
import CarlaAutoParking.utils.plt_utils as plt_utils
import numpy as np
import carla
import pygame
from CarlaAutoParking.planner.serach_utils import (
    downsample_smooth,
    upsample_interpolation,
    downup_smooth,
)
from CarlaAutoParking.controller.carla_controller import (
    CarlaCruiseController,
    CarlaParkingController,
)

import time

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w

except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, player, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            player.set_light_state(self._lights)
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0

    def parse_events(self, player: carla.Vehicle, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:  # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:  # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if (
                    current_lights != self._lights
                ):  # Change the light state only if necessary
                    self._lights = current_lights
                    player.set_light_state(carla.VehicleLightState(self._lights))
            player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 0.2)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


def test_keyboard_control():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    sim_world: carla.World = client.load_world("Town05_Opt")
    # sim_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    dt = 0.02
    settings = sim_world.get_settings()
    settings.substepping = True
    settings.max_substep_delta_time = dt / 16
    settings.max_substeps = 16
    settings.fixed_delta_seconds = dt
    settings.synchronous_mode = True

    sim_world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    ego: carla.Vehicle = carla_utils.spawn_parking_entrance_ego_car(sim_world)
    # ego.set_autopilot(True)

    pygame.init()
    pygame.font.init()

    try:
        display = pygame.display.set_mode(
            (400, 400), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        display.fill((0, 0, 0))
        pygame.display.flip()
        controller = KeyboardControl(ego, False)
        sim_world.tick()
        clock = pygame.time.Clock()

        tracking_path = []

        while True:
            sim_world.tick()
            clock.tick_busy_loop(60)
            carla_utils.set_spectator(ego, 30)
            current_se2state = carla_utils.generate_rear_se2state_from_vehicle(ego)

            tracking_path += [current_se2state]
            carla_utils.draw_path(sim_world.debug, tracking_path)
            if controller.parse_events(ego, clock):
                return

    finally:
        ego.destroy()
        pygame.quit()


def test_controller():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    sim_world: carla.World = client.load_world("Town05_Opt")
    # sim_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    dt = 0.02
    settings = sim_world.get_settings()
    settings.substepping = True
    settings.max_substep_delta_time = dt / 16
    settings.max_substeps = 16
    settings.fixed_delta_seconds = dt
    settings.synchronous_mode = True

    sim_world.apply_settings(settings)

    ego: carla.Vehicle = carla_utils.spawn_parking_entrance_ego_car(sim_world)
    ego.apply_control(carla.VehicleControl())
    sim_world.tick()

    load_trajectory = plt_utils.load_pickle_file(
        "ParkingSimulation/park_trajectory.pickle"
    )
    N = len(load_trajectory)

    reference_trajectory = load_trajectory[50:]
    chassis_controller = CarlaParkingController()
    chassis_controller.set_reference_trajectory(reference_trajectory)

    N = len(reference_trajectory)
    tt = np.linspace(0, N * dt, N)

    current_rear_se2state = carla_utils.generate_rear_se2state_from_vehicle(ego)
    current_rear_se2state.t = 0.0
    carla_utils.set_spectator(ego)
    tracking_rear_trajectory, tracking_center_trajectory = [], []
    time_count = 0

    while True:
        sim_world.tick()
        time_count += 1
        carla_utils.set_spectator(ego, z=20)
        current_rear_se2state = carla_utils.generate_rear_se2state_from_vehicle(ego)
        current_center_se2state = carla_utils.generate_center_se2state_from_vehicle(ego)
        current_rear_se2state.t = time_count * dt
        current_center_se2state.t = time_count * dt
        control = chassis_controller.action(current_rear_se2state)
        ego.apply_control(control)
        tracking_rear_trajectory += [current_rear_se2state]

        tracking_center_trajectory += [current_center_se2state]
        carla_utils.draw_path(
            sim_world.debug, reference_trajectory, color=(0, 255, 0), gap=5
        )

    # plt_utils.save_pickle_file(tracking_center_trajectory,"tracking_path.pickle")


if __name__ == "__main__":
    # test_keyboard_control()
    test_controller()
