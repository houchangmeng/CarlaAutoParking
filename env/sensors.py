import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent) + "/dist")


import carla
import numpy as np
from carla import Actor, Transform, Location, Rotation
from carla import ColorConverter as cc
from typing import Dict, List, Tuple
import weakref
from queue import Queue
import cv2
import math


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.imu")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data)
        )
        self.data_q: Queue = Queue()

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)),
        )
        self.gyroscope = (
            max(limits[0], min(limits[1], sensor_data.gyroscope.x)),
            max(limits[0], min(limits[1], sensor_data.gyroscope.y)),
            max(limits[0], min(limits[1], sensor_data.gyroscope.z)),
        )

        self.compass = math.degrees(sensor_data.compass)
        imudata = (self.accelerometer, self.gyroscope, self.compass)
        self.data = imudata
        self.data_q.put(imudata)


class CameraSensor:
    def __init__(self, parent_actor: Actor) -> None:
        self.data = None
        self.parent = parent_actor
        self.world = parent_actor.get_world()
        blueprint_library = self.world.get_blueprint_library()
        self.bp = blueprint_library.find("sensor.camera.rgb")
        self.sensor = None
        self.data_q: Queue = Queue()

    def set_sensor(self, sensor_attributes: Dict, attach_transform: carla.Transform):
        for key in sensor_attributes:
            self.bp.set_attribute(key, sensor_attributes[key])
            self.bp.set_attribute(key, sensor_attributes[key])

        self.sensor = self.world.spawn_actor(
            self.bp, attach_transform, attach_to=self.parent
        )
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.data = array
        self.data_q.put(array)


class CamerasManager:
    """
    Reference camera parameter https://www.eetimes.com/a-tesla-model-3-tear-down-after-a-hardware-retrofit/2/.
    """

    def __init__(self, player: Actor) -> None:
        """
        Front
        """
        self.front = CameraSensor(player)
        front_attr = {"image_size_x": "1280", "image_size_y": "960", "fov": "150"}
        front_trans = carla.Transform(
            carla.Location(x=2.37, y=0, z=1.3), carla.Rotation(pitch=0, yaw=0)
        )
        self.front.set_sensor(front_attr, front_trans)

        """
        Back
        """
        self.back = CameraSensor(player)
        back_attr = {"image_size_x": "1280", "image_size_y": "960", "fov": "120"}
        back_trans = carla.Transform(
            carla.Location(x=-2.37, y=0, z=0.74), carla.Rotation(pitch=0, yaw=180)
        )
        self.back.set_sensor(back_attr, back_trans)

        """
        Left front
        """
        self.lf = CameraSensor(player)
        lf_attr = {"image_size_x": "320", "image_size_y": "320", "fov": "90"}
        lf_trans = carla.Transform(
            carla.Location(x=-0.1, y=-0.9, z=1.2), carla.Rotation(pitch=0, yaw=-70)
        )
        self.lf.set_sensor(lf_attr, lf_trans)

        """
        Left back
        """
        self.lb = CameraSensor(player)
        lb_attr = {"image_size_x": "320", "image_size_y": "320", "fov": "80"}
        lb_trans = carla.Transform(
            carla.Location(x=0.8, y=-1.0, z=0.74), carla.Rotation(pitch=0, yaw=-150)
        )
        self.lb.set_sensor(lb_attr, lb_trans)

        """
        Right front
        """
        self.rf = CameraSensor(player)
        rf_attr = {"image_size_x": "320", "image_size_y": "320", "fov": "90"}
        rf_trans = carla.Transform(
            carla.Location(x=-0.1, y=0.9, z=1.2), carla.Rotation(pitch=0, yaw=70)
        )
        self.rf.set_sensor(rf_attr, rf_trans)

        """
        Right Back
        """
        self.rb = CameraSensor(player)
        rb_attr = {"image_size_x": "320", "image_size_y": "320", "fov": "80"}
        rb_trans = carla.Transform(
            carla.Location(x=0.8, y=1.0, z=0.74), carla.Rotation(pitch=0, yaw=150)
        )
        self.rb.set_sensor(rb_attr, rb_trans)

        """
        BEV
        """
        self.bev = CameraSensor(player)
        bev_attr = {"image_size_x": "620", "image_size_y": "620", "fov": "90"}
        bev_trans = carla.Transform(
            carla.Location(x=0, z=8), carla.Rotation(pitch=-90, yaw=0)
        )
        self.bev.set_sensor(bev_attr, bev_trans)

    @property
    def rawdata(self):
        return {
            "front": self.front.data_q.get(),
            "back": self.back.data_q.get(),
            "lf": self.lf.data_q.get(),
            "lb": self.lb.data_q.get(),
            "rf": self.rf.data_q.get(),
            "rb": self.rb.data_q.get(),
            "bev": self.bev.data_q.get(),
        }


def main():
    import matplotlib.pyplot as plt
    import CarlaAutoParking.env.carla_utils as carla_utils

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    sim_world: carla.World = client.load_world("Town05_Opt")

    settings = sim_world.get_settings()
    settings.fixed_delta_seconds = 0.02
    settings.synchronous_mode = True
    sim_world.apply_settings(settings)

    sim_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    carla_utils.set_parking_spectator(sim_world)
    carla_utils.set_parking_obstacles(sim_world)

    player = carla_utils.spawn_parking_ego_car(sim_world)
    cam_manager = CamerasManager(player)

    try:
        while True:
            sim_world.tick()
            img = carla_utils.render_image(cam_manager.rawdata)
            cv2.imshow("Playback", img[:, :, ::-1])
            cv2.waitKey(1)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

    finally:
        carla_utils.destory_all_actors(sim_world)


if __name__ == "__main__":
    main()
