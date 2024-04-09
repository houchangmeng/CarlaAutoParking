import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent) + "/dist")

import numpy as np
import matplotlib.pyplot as plt
from CarlaAutoParking.others.geometry import move_vertexes_array, ndarray_to_vertexlist
import carla
from carla import Actor
from CarlaAutoParking.others.se2state import SE2, SE2State
from typing import List


def draw_vertexes_list(debug: carla.DebugHelper, vertexes_list):
    for point in vertexes_list:
        point = carla.Vector3D(point[0], point[1], 0.2)
        debug.draw_point(point, life_time=2)


def draw_vertexes_array(debug: carla.DebugHelper, vertexes_array):
    vertexes_list = ndarray_to_vertexlist(vertexes_array)
    draw_vertexes_list(debug, vertexes_list)


def draw_se2state_point(
    debug: carla.DebugHelper,
    se2state: SE2State,
    color=(255, 0, 0),
):
    draw_color = carla.Color(color[0], color[1], color[2])
    point = carla.Vector3D(se2state.se2.x, se2state.se2.y, 0.4)
    debug.draw_point(point, life_time=1.0, color=draw_color)


def draw_path(
    debug: carla.DebugHelper,
    global_path: List[SE2State],
    color=(255, 0, 0),
    thickness=0.05,
    gap: int = None,
):
    if len(global_path) < 10:
        return
    draw_color = carla.Color(color[0], color[1], color[2])
    if gap is None:
        N = len(global_path)
        gap = int(N / 30) + 1

    path = global_path[::gap]

    n = len(path)

    for i in range(n - 1):
        begin = carla.Location(path[i].se2.x, path[i].se2.y, 0.3)
        end = carla.Location(path[i + 1].se2.x, path[i + 1].se2.y, 0.3)
        debug.draw_arrow(
            begin, end, life_time=0.1, color=draw_color, thickness=thickness
        )


import cv2


def render_image(rawdata):
    lf_img = rawdata["lf"]
    lb_img = rawdata["lb"]
    L_img = np.concatenate([np.fliplr(lf_img), lb_img], axis=1).swapaxes(0, 1)

    rf_img = rawdata["rf"]
    rb_img = rawdata["rb"]
    R_img = np.concatenate([rf_img, rb_img], axis=1).swapaxes(0, 1)
    R_img = np.rot90(R_img, 2)

    bev_img = rawdata["bev"]

    BEV_img = cv2.copyMakeBorder(
        bev_img,
        10,
        10,
        10,
        10,
        cv2.BORDER_CONSTANT,
        value=[0, 255, 0],
    )

    M_img = np.concatenate([L_img, BEV_img, R_img], axis=1)

    white_block = np.ones((320, 320, 3), dtype=np.uint8) * 255

    front_img = rawdata["front"]
    front_img = cv2.resize(front_img, (640, 320))
    F_img = np.concatenate([white_block, front_img, white_block], axis=1)

    back_img = rawdata["back"]
    back_img = cv2.resize(back_img, (640, 320))
    B_img = np.concatenate([white_block, back_img, white_block], axis=1)

    camera_perception = np.concatenate([F_img, M_img, B_img], axis=0)

    return camera_perception


def get_vertexes_array_from_actor(actor: carla.Actor):
    extent_x = actor.bounding_box.extent.x
    extent_y = actor.bounding_box.extent.y

    trans = actor.get_transform()

    forward_vec = trans.get_forward_vector()
    rot_angle = np.arctan2(forward_vec.y, forward_vec.x)

    # rot_angle = np.deg2rad(trans.rotation.yaw)
    boundingbox_array = np.array(
        [
            [-extent_x, -extent_y],
            [extent_x, -extent_y],
            [extent_x, extent_y],
            [-extent_x, extent_y],
        ]
    ).T

    rect_center = np.array([[trans.location.x], [trans.location.y]])

    rect_array = move_vertexes_array(boundingbox_array, rot_angle, rect_center)

    return rect_array


def get_boundingbox_polygon_from_actor(actor: carla.Actor):
    boundingbox_array = get_vertexes_array_from_actor(actor)
    boundingbox_vertexes_list = ndarray_to_vertexlist(boundingbox_array)
    return boundingbox_vertexes_list


def set_parking_spectator(world: carla.World):
    spectator = world.get_spectator()

    spectator_transform = carla.Transform(
        carla.Location(x=-1.8, y=-27, z=16),
        carla.Rotation(pitch=-90, yaw=180, roll=-90),
    )
    spectator.set_transform(spectator_transform)


def set_parking_obstacles(world: carla.World, obstacle_num=12):
    import random

    blueprint_library = world.get_blueprint_library()
    rot = carla.Rotation(0, -0, 0)
    obstacle_transforms = []
    for i in range(6):
        loc = carla.Location(3.2, -18.8 - 2.8 * i, 0.3)
        obstacle_transforms += [carla.Transform(loc, rot)]
        loc = carla.Location(-6.8, -18.8 - 2.8 * i, 0.3)
        obstacle_transforms += [carla.Transform(loc, rot)]

    # N_obs = len(obstacle_transforms) - 4
    N_obs = obstacle_num

    obstacle_actor_list = []

    while len(obstacle_actor_list) < N_obs:
        # obstacle_bp = random.choice(blueprint_library.filter("vehicle.audi.*"))
        obstacle_bp = random.choice(blueprint_library.filter("vehicle.tesla.model3"))
        color_str = (
            str(random.randint(0, 200))
            + ","
            + str(random.randint(0, 255))
            + ","
            + str(random.randint(0, 255))
        )

        obstacle_bp.set_attribute("color", color_str)
        transform = random.choice(obstacle_transforms)
        obstacle_car = world.try_spawn_actor(obstacle_bp, transform)
        if obstacle_car is not None:
            obstacle_actor_list += [obstacle_car]
            obstacle_transforms.remove(transform)

    target_transform_list = obstacle_transforms

    return obstacle_actor_list, target_transform_list


import random


def set_town5_parking_lot(world: carla.World, parking_slot_num=12):
    blueprint_library = world.get_blueprint_library()
    parking_slot_transforms = []

    for i in range(10):
        rot = carla.Rotation(0, -0, 0)
        loc = carla.Location(9.4, -18.8 - 2.8 * i, 0.5)
        parking_slot_transforms += [carla.Transform(loc, rot)]

        rot = carla.Rotation(0, 180, 0)
        loc = carla.Location(3.2, -18.8 - 2.8 * i, 0.5)
        parking_slot_transforms += [carla.Transform(loc, rot)]

        rot = carla.Rotation(0, 0, 0)
        loc = carla.Location(-6.8, -18.8 - 2.8 * i, 0.5)
        parking_slot_transforms += [carla.Transform(loc, rot)]

        rot = carla.Rotation(0, 180, 0)
        loc = carla.Location(-13.0, -18.8 - 2.8 * i, 0.5)
        parking_slot_transforms += [carla.Transform(loc, rot)]

        rot = carla.Rotation(0, 0, 0)
        loc = carla.Location(-23.0, -18.8 - 2.8 * i, 0.5)
        parking_slot_transforms += [carla.Transform(loc, rot)]

        rot = carla.Rotation(0, 180, 0)
        loc = carla.Location(-29.2, -18.8 - 2.8 * i, 0.5)
        parking_slot_transforms += [carla.Transform(loc, rot)]

    num_slots = len(parking_slot_transforms)
    num_obs = num_slots - parking_slot_num
    obstacle_actor_list = []

    while len(obstacle_actor_list) < num_obs:
        obstacle_bp = random.choice(blueprint_library.filter("vehicle.audi.*"))
        # obstacle_bp = random.choice(blueprint_library.filter("vehicle.tesla.model3"))
        color_str = (
            str(random.randint(0, 200))
            + ","
            + str(random.randint(0, 255))
            + ","
            + str(random.randint(0, 255))
        )

        obstacle_bp.set_attribute("color", color_str)
        transform = random.choice(parking_slot_transforms)
        obstacle_car = world.try_spawn_actor(obstacle_bp, transform)
        if obstacle_car is not None:
            obstacle_actor_list += [obstacle_car]
            parking_slot_transforms.remove(transform)

    target_transform_list = parking_slot_transforms
    return obstacle_actor_list, target_transform_list


def get_carla_transform_from_se2(se2: SE2):
    location = carla.Location(se2.x, se2.y, 0.3)  # ego_car start
    rotation = carla.Rotation(
        0, np.rad2deg(se2.so2.heading), 0
    )  # ego_car start (0, -90,0)
    transform = carla.Transform(location, rotation)

    return transform


def get_se2_from_carla_transform(transform: carla.Transform):
    x = transform.location.x
    y = transform.location.y
    yaw = np.deg2rad(transform.rotation.yaw)

    return SE2(x, y, yaw)


def spawn_ego_from_carla_transform(world: carla.World, transform: carla.Transform):
    spawn_transform = transform
    blueprint_library = world.get_blueprint_library()
    ego_bp = blueprint_library.find("vehicle.tesla.model3")

    ego: carla.Vehicle = world.spawn_actor(ego_bp, spawn_transform)
    physics_control = ego.get_physics_control()

    physics_control.gear_switch_time = 0.0

    physics_control.damping_rate_full_throttle = 0
    physics_control.damping_rate_zero_throttle_clutch_engaged = 0
    physics_control.damping_rate_zero_throttle_clutch_disengaged = 0

    physics_control.torque_curve = [
        carla.Vector2D(x=0, y=800),
        carla.Vector2D(x=10000, y=800),
    ]
    physics_control.steering_curve = [
        carla.Vector2D(x=0.000000, y=1.000000),
        carla.Vector2D(x=100.000000, y=1.000000),
    ]
    physics_control.mass = 1752
    physics_control.drag_coefficient = 0.0

    # Location(x=0.450000, y=0.000000, z=-0.300000)
    # physics_control.center_of_mass = carla.Location(x=0.40000, y=0.000000, z=-0.300000)

    fl_wheel_control = physics_control.wheels[0]
    fr_wheel_control = physics_control.wheels[1]
    rl_wheel_control = physics_control.wheels[2]
    rr_wheel_control = physics_control.wheels[3]

    fl_wheel_control.max_steer_angle = 57
    fr_wheel_control.max_steer_angle = 57

    physics_control.wheels = [
        fl_wheel_control,
        fr_wheel_control,
        rl_wheel_control,
        rr_wheel_control,
    ]

    ego.apply_physics_control(physics_control)

    return ego


def spawn_ego_from_se2(world: carla.World, spawn_se2: SE2):
    spawn_transform = get_carla_transform_from_se2(spawn_se2)

    ego = spawn_ego_from_carla_transform(world, spawn_transform)

    return ego


from config import ParkingConfig
from copy import deepcopy


def get_rear_se2state_from_center(center_se2state: SE2State) -> SE2State:
    offset = 1.38
    rear_se2state = deepcopy(center_se2state)
    heading = center_se2state.heading
    rear_se2state.se2.x = center_se2state.se2.x - offset * np.cos(heading)
    rear_se2state.se2.y = center_se2state.se2.y - offset * np.sin(heading)
    return rear_se2state


def get_center_se2state_from_rear(rear_se2state: SE2State) -> SE2State:
    offset = 1.38
    center_se2state = deepcopy(rear_se2state)
    heading = rear_se2state.heading
    center_se2state.se2.x = rear_se2state.se2.x + offset * np.cos(heading)
    center_se2state.se2.y = rear_se2state.se2.y + offset * np.sin(heading)
    return center_se2state


def generate_center_se2state_list_from_rear_list(
    rear_se2state_list: List[SE2State],
) -> List[SE2State]:
    center_se2state_list = []
    for rear_se2state in rear_se2state_list:
        center_se2state_list += [get_center_se2state_from_rear(rear_se2state)]

    return center_se2state_list


def generate_rear_se2state_list_from_center_list(
    center_se2state_list: List[SE2State],
) -> List[SE2State]:
    rear_se2state_list = []
    for center_se2state in center_se2state_list:
        rear_se2state_list += [get_rear_se2state_from_center(center_se2state)]

    return rear_se2state_list


def generate_parking_vertexes_from_carla_location_se2(
    parkingse2: SE2, parking_cfg: ParkingConfig = ParkingConfig()
):
    """
    Pay attention to here, is parking transform is center location.
    """
    parking_array = parking_cfg.parking_array
    rot_angle = parkingse2.so2.heading
    xy_offset = np.array([[parkingse2.x], [parkingse2.y]])
    parking_array = move_vertexes_array(parking_array, rot_angle, xy_offset)
    parking_vertexes = ndarray_to_vertexlist(parking_array)

    return parking_vertexes


def generate_perception_from_cheating(
    obstacle_actor_list: List[Actor],
    parking_carla_transform_list: List[carla.Transform],
):
    obstacle_vertexes_se2_list = []

    for actor in obstacle_actor_list:
        trans = actor.get_transform()
        forward_vec = trans.get_forward_vector()
        # heading = np.arctan2(forward_vec.y, forward_vec.x)

        obstacle_vertexes_se2_list += [
            ndarray_to_vertexlist(get_vertexes_array_from_actor(actor))
        ]

    parking_vertexes_se2_list = []

    for transform in parking_carla_transform_list:
        """
        Pay attention to here, parking transform is center location.
        """
        parking_se2 = SE2(
            transform.location.x, transform.location.y, transform.rotation.yaw
        )
        parking_vertexes_se2_list += [
            generate_parking_vertexes_from_carla_location_se2(parking_se2)
        ]

    return obstacle_vertexes_se2_list, parking_vertexes_se2_list


def spawn_parking_ego_car(world: carla.World) -> carla.Vehicle:
    spawn_location = carla.Location(-1.5, -25.2, 0.3)  # ego_car start
    spawn_rotation = carla.Rotation(0, 100, 0)  # ego_car start (0, -90,0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)

    return spawn_ego_from_carla_transform(world, spawn_transform)


def spawn_calibration_ego_car(world: carla.World) -> carla.Vehicle:
    spawn_location = carla.Location(-33.4, 6.18, 0.1)  # ego_car start
    spawn_rotation = carla.Rotation(0, 0, 0)  # ego_car start (0, -90,0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_ego_from_carla_transform(world, spawn_transform)


def spawn_parking_entrance_ego_car(world: carla.World) -> carla.Vehicle:
    spawn_location = carla.Location(-34.66, -29.04, 0.5)  # ego_car start
    spawn_rotation = carla.Rotation(0, 90, 0)  # ego_car start (0, -90,0)
    spawn_transform = carla.Transform(spawn_location, spawn_rotation)
    return spawn_ego_from_carla_transform(world, spawn_transform)


def set_spectator(vehicle: carla.Vehicle, z=10):
    transform = vehicle.get_transform()

    world = vehicle.get_world()
    spec = world.get_spectator()

    spec.set_transform(
        carla.Transform(
            transform.location + carla.Location(z=z),
            carla.Rotation(yaw=180, pitch=-90),
        )
    )


def get_next_waypoint(vehicle: carla.Vehicle, distance=0.1) -> carla.Waypoint:
    sim_map = vehicle.get_world().get_map()
    waypoint = sim_map.get_waypoint(vehicle.get_location())
    next_waypoint = waypoint.next(distance)[0]
    next_waypoint.transform.location.z = 1.0

    return next_waypoint


def generate_rear_se2state_from_vehicle(vehicle: carla.Vehicle) -> SE2State:
    vehicle_trans = vehicle.get_transform()
    vehicle_acc = vehicle.get_acceleration()
    vehicle_vel = vehicle.get_velocity()
    vehicle_loc = vehicle_trans.location
    vehicle_heading = np.deg2rad(vehicle_trans.rotation.yaw)

    forward_vec = vehicle_trans.get_forward_vector()
    # forward_vec = np.array([np.cos(vehicle_heading), np.sin(vehicle_heading)])
    # forward_vec = carla.Vector3D(np.cos(vehicle_heading), np.sin(vehicle_heading), 0)
    vehicle_acc = vehicle_acc.dot(forward_vec)
    vehicle_vel = vehicle_vel.dot(forward_vec)

    fl_delta = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
    fr_delta = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)

    ego_delta = 0.5 * (fl_delta + fr_delta) / 180 * np.pi

    offset = 1.38
    vehicle_x = vehicle_loc.x - offset * forward_vec.x
    vehicle_y = vehicle_loc.y - offset * forward_vec.y

    vehicle_se2 = SE2(vehicle_x, vehicle_y, vehicle_heading)
    vehicle_se2state = SE2State.from_se2(vehicle_se2)
    vehicle_se2state.a = vehicle_acc
    vehicle_se2state.v = vehicle_vel
    vehicle_se2state.delta = ego_delta

    return vehicle_se2state


def generate_center_se2state_from_vehicle(vehicle: carla.Vehicle) -> SE2State:
    vehicle_trans = vehicle.get_transform()
    vehicle_acc = vehicle.get_acceleration()
    vehicle_vel = vehicle.get_velocity()
    vehicle_loc = vehicle_trans.location
    vehicle_heading = np.deg2rad(vehicle_trans.rotation.yaw)

    # print(vehicle_acc)

    forward_vec = vehicle_trans.get_forward_vector()

    vehicle_acc = vehicle_acc.dot(forward_vec)
    vehicle_vel = vehicle_vel.dot(forward_vec)

    if abs(vehicle_acc) > 10:
        vehicle_acc = np.sign(vehicle_acc) * 10
    fl_delta = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
    fr_delta = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)

    ego_delta = 0.5 * (fl_delta + fr_delta) / 180 * np.pi

    # offset = 1.38
    # vehicle_x = vehicle_loc.x  # - offset * forward_vec.x
    # vehicle_y = vehicle_loc.y  # - offset * forward_vec.y

    vehicle_se2 = SE2(vehicle_loc.x, vehicle_loc.y, vehicle_heading)
    vehicle_se2state = SE2State.from_se2(vehicle_se2)
    vehicle_se2state.a = vehicle_acc
    vehicle_se2state.v = vehicle_vel
    vehicle_se2state.delta = ego_delta

    return vehicle_se2state


def generate_se2state_from_waypoint(waypoint: carla.Waypoint):
    waypoint_trans = waypoint.transform

    waypoint_loc = waypoint_trans.location
    waypoint_heading = np.deg2rad(waypoint_trans.rotation.yaw)

    waypoint_se2 = SE2(waypoint_loc.x, waypoint_loc.y, waypoint_heading)
    waypoint_se2state = SE2State.from_se2(waypoint_se2)
    waypoint_se2state.v = 2
    waypoint_se2state.a = 0
    waypoint_se2state.delta = 0

    return waypoint_se2state


def destory_all_actors(world: carla.World):
    actor_list = world.get_actors()
    for actor in actor_list:
        if actor.is_alive:
            actor.destroy()


import math


def get_camera_intrinsic_transform(fov_x=90, pixel_width=640, pixel_height=640):
    fov_y = pixel_height / pixel_width * fov_x

    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = (pixel_width / 2.0) / np.tan(np.deg2rad(fov_x / 2))
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = (pixel_height / 2.0) / np.tan(np.deg2rad(fov_y / 2))
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics


def camera_intrinsic_transform(fov_x=90, fov_y=60, pixel_width=640, pixel_height=640):
    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = (pixel_width / 2.0) / np.tan(np.deg2rad(fov_x / 2.0))
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = (pixel_height / 2.0) / np.tan(np.deg2rad(fov_y / 2.0))
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics


def camera_intrinsic_fov(intrinsic):
    # 计算FOV
    w, h = intrinsic[0][2] * 2, intrinsic[1][2] * 2
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    # Go
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))
    return fov_x, fov_y
