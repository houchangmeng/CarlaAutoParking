import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from typing import Optional, Tuple, Union, List, Dict
from CarlaAutoParking.others.geometry import Polygon
from CarlaAutoParking.others.se2state import SE2State


import cv2
import carla
import gym
import numpy as np
import CarlaAutoParking.env.carla_utils as carla_utils
import CarlaAutoParking.others.coord_transform as ct
import CarlaAutoParking.utils.plt_utils as plt_utils
from sensors import CamerasManager, IMUSensor, CameraSensor
from config import CarlaEnvConfig, DataCollectionEnvConfig
import time


class DataCollectionEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(
        self,
        env_cfg=DataCollectionEnvConfig(),
    ) -> None:
        super().__init__()

        """
        Environment.
        """
        self.env_cfg = env_cfg
        self.sim_world = None
        self.client = None
        self.ego = None
        self.perception_info = None
        self.dt = self.env_cfg.dt
        self.start_global_se2 = None

        """
        Sensors
        """
        self.camera_manager: CamerasManager = None
        self.raw_cameradata = None
        self.bev = None
        self.bev_img = None

        """
        Parking Task.
        """
        self.obstacle_actor_list = None
        self.target_transform_list = None
        self.goal_se2state = None

        """
        Ego state
        """
        self.local_se2state: SE2State = None
        self.global_se2state: SE2State = None
        self.global_veh_center_ref_traj = None
        self.global_veh_center_tracking_traj = None

        """
        Others.
        """
        self.record_image_list = None
        self.current_time = None
        self.step_count = None
        self.isopen = True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)

        if self.sim_world is not None:
            carla_utils.destory_all_actors(self.sim_world)

        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(20.0)

        self.sim_world: carla.World = self.client.load_world("Town05_Opt")

        """
        Set synchronous mode.
        """

        settings = self.sim_world.get_settings()
        settings.substepping = True
        settings.max_substep_delta_time = self.dt / 16
        settings.max_substeps = 16
        settings.fixed_delta_seconds = self.dt
        settings.synchronous_mode = True
        settings.synchronous_mode = True
        self.sim_world.apply_settings(settings)

        """
        Remove redundant vehicles.
        """
        self.sim_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        """
        Set ego car.
        """
        self.ego = carla_utils.spawn_ego_from_se2(
            self.sim_world, self.env_cfg.spawn_point
        )

        control = carla.VehicleControl()
        control.throttle = 0.02
        for i in range(100):
            self.ego.apply_control(control)
            self.sim_world.tick()

        (
            self.obstacle_actor_list,
            self.target_transform_list,
        ) = carla_utils.set_town5_parking_lot(
            self.sim_world, self.env_cfg.parking_slot_num
        )
        """
        Set cameras.
        """

        if self.env_cfg.enable_all_camera:
            self.camera_manager = CamerasManager(self.ego)

        if self.env_cfg.enable_bev_camera:
            self.bev = CameraSensor(self.ego)
            bev_attr = {"image_size_x": "640", "image_size_y": "640", "fov": "90"}
            bev_trans = carla.Transform(
                carla.Location(x=0, z=8), carla.Rotation(pitch=-90, yaw=0)
            )
            self.bev.set_sensor(bev_attr, bev_trans)

        if self.env_cfg.enable_imu:
            self.imu = IMUSensor(self.ego)

        """
        Set recorder.
        """

        self.record_image_list = []
        """
        Set IMU.
        """
        # TODO

        """
        Set initial se2state.
        """
        self.sim_world.tick()

        global_se2state = carla_utils.generate_center_se2state_from_vehicle(self.ego)
        self.global_se2state = global_se2state

        self.start_global_se2 = global_se2state.se2

        local_se2state = ct.change_se2state_coord(
            global_se2state, self.start_global_se2
        )

        self.local_se2state = local_se2state
        self.global_veh_center_tracking_traj = []
        self.global_veh_center_ref_traj = []

        self.current_time = 0.0
        self.step_count = 0.0

        carla_utils.set_spectator(self.ego)

        return (
            self.local_se2state,
            {"camera_data": None, "raw_imu_data": None},
        )

    def step(
        self, action: carla.VehicleControl
    ) -> Tuple[SE2State, float, bool, bool, dict]:
        self.sim_world.tick()
        """
        Perception and localization.
        """

        camera_data, imu_data = self.perception()

        self.local_se2state, self.global_se2state = self.localization()
        self.local_se2state.t = self.step_count * self.dt
        self.global_se2state.t = self.step_count * self.dt

        """
        Apply control.
        """
        self.ego.apply_control(action)

        """
        Is Done?
        """
        reward = 1.0
        terminated = False
        if len(self.global_veh_center_ref_traj) > 0:
            done = self.goal_se2state.is_close_enough(self.global_se2state)
            self.global_veh_center_tracking_traj += [self.global_se2state]
        else:
            done = False

        """
        Render
        """
        self.render()

        self.step_count += 1

        self.debug()
        return (
            self.local_se2state,
            reward,
            terminated,
            done,
            {"camera_data": camera_data, "raw_imu_data": imu_data},
        )

    def set_tracking_task(
        self,
        global_center_ref_traj: List[SE2State],
    ):
        self.global_veh_center_ref_traj = global_center_ref_traj
        self.goal_se2state = global_center_ref_traj[-1]
        self.global_veh_center_tracking_traj = []

    def debug(self):
        carla_utils.draw_path(
            self.sim_world.debug, self.global_veh_center_ref_traj, (255, 0, 0)
        )
        carla_utils.draw_path(
            self.sim_world.debug,
            self.global_veh_center_tracking_traj,
            (0, 255, 0),
            thickness=0.1,
        )
        carla_utils.set_spectator(self.ego)

    def perception(self):
        """
        TODO: generate perception info from camera.
        """

        if self.env_cfg.enable_all_camera:
            camera_data = self.camera_manager.rawdata
        elif self.env_cfg.enable_bev_camera:
            camera_data = self.bev.data_q.get()
        else:
            camera_data = None

        if self.env_cfg.enable_imu:
            imu_data = self.imu.data_q.get()
        else:
            imu_data = None

        return camera_data, imu_data

    def perception_todo(self, camera_data):
        obstacle_polygon_list_in_current_se2 = []
        parking_polygon_list_in_current_se2 = []
        obstacle_polygon_list_in_global_se2 = []
        parking_polygon_list_in_global_se2 = []

        obstacle_vertexes_list_in_vehicle_coord = self.obstacle_vertexes_recognition(
            camera_data
        )
        parking_vertexes_list_in_vehicle_coord = self.parking_vertexes_recognition(
            camera_data
        )

        for obstacle_vertexes in obstacle_vertexes_list_in_vehicle_coord:
            obstacle_polygon_list_in_current_se2 += [Polygon(obstacle_vertexes)]

        for parking_vertexes in parking_vertexes_list_in_vehicle_coord:
            parking_polygon_list_in_current_se2 += [Polygon(parking_vertexes)]

        return (
            obstacle_polygon_list_in_current_se2,
            parking_polygon_list_in_current_se2,
            obstacle_polygon_list_in_global_se2,
            parking_polygon_list_in_global_se2,
        )

    def parking_vertexes_recognition(self, img=np.ndarray):
        raise NotImplementedError(
            "parking_vertexes_recognition carla environment function does not implement."
        )

    def obstacle_vertexes_recognition(self, img=np.ndarray):
        raise NotImplementedError(
            "obstacle_vertexes_recognition carla environment function does not implement."
        )

    def localization(self, image_data=None, imu_data=None):
        global_se2state = carla_utils.generate_center_se2state_from_vehicle(self.ego)
        local_se2state = ct.change_se2state_coord(
            global_se2state, self.start_global_se2
        )
        return local_se2state, global_se2state

    def render(self):
        from copy import deepcopy

        if self.env_cfg.enable_all_camera:
            img = carla_utils.render_image(self.camera_manager.rawdata)
        elif self.env_cfg.enable_bev_camera:
            img = deepcopy(self.bev.data)
        else:
            img = None

        if img is None:
            return

        if self.env_cfg.record:
            self.record_image_list += [img]

        if self.env_cfg.render_mode == "human":
            cv2.imshow("Playback", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # cv2.imshow("Playback", img)
            cv2.waitKey(1)  # 20 ms

        else:
            return img

    def close(self):
        if self.sim_world is not None:
            carla_utils.destory_all_actors(self.sim_world)

        if self.env_cfg.record and len(self.record_image_list) > 10:
            fps = int(1 / self.dt)

            folder = str(pathlib.Path(__file__).parent.parent) + "/videos/"
            file_name = folder + str(time.time_ns()) + ".mp4"
            plt_utils.image_list_to_video(self.record_image_list, file_name, fps)

        if self.env_cfg.render_mode == "human":
            cv2.destroyAllWindows()

        self.isopen = False


import matplotlib.pyplot as plt


def main():
    env_cfg = DataCollectionEnvConfig()
    env_cfg.render_mode = "human"
    env_cfg.enable_bev_camera = True
    env_cfg.record = False
    env = DataCollectionEnv(env_cfg)
    env.reset()

    plt.figure(0, figsize=[8, 8])
    ax = plt.gca()
    ax.invert_xaxis()

    N = 200

    try:
        while True:
            action = carla.VehicleControl()
            action.throttle = 0.1
            se2state, _, _, _, info = env.step(action)
            # env.render()
            # plt.cla()
            # plt_utils.plot_task(info["local_obstacle_polygon_list"], se2state, se2state)
            # plt.draw()
            # plt.pause(0.01)
    finally:
        env.close()


if __name__ == "__main__":
    main()
