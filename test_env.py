import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))


import CarlaAutoParking.utils.plt_utils as plt_utils
import CarlaAutoParking.env.carla_utils as carla_utils
from CarlaAutoParking.env.parkinglot_datacollection_carla_env import DataCollectionEnv
from CarlaAutoParking.controller.carla_controller import CarlaParkingController
from CarlaAutoParking.others.coord_transform import change_se2state_list_coord
from config import DataCollectionEnvConfig


def main():
    global_center_ref_traj = plt_utils.load_pickle_file(
        "ParkingSimulation/tracking_path.pickle"
    )
    spawn_point = global_center_ref_traj[0].se2

    env_cfg = DataCollectionEnvConfig()
    env_cfg.parking_slot_num = 15
    env_cfg.enable_bev_camera = True
    env_cfg.record = True
    env_cfg.spawn_point = spawn_point
    env = DataCollectionEnv(env_cfg)

    current_center_se2state, info = env.reset()

    local_center_ref_traj = change_se2state_list_coord(
        global_center_ref_traj, global_center_ref_traj[0]
    )
    controller = CarlaParkingController()
    ref_trajectory = local_center_ref_traj[100:]
    controller.set_reference_trajectory(ref_trajectory)
    N = len(ref_trajectory)
    for i in range(N):
        control = controller.action(current_center_se2state)
        current_center_se2state, _, _, done, info = env.step(control)
    env.close()


if __name__ == "__main__":
    main()
