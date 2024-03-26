import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from typing import TypeVar, List
import CarlaAutoParking.utils.plt_utils as plt_utils
import CarlaAutoParking.utils.parking_utils as st
from CarlaAutoParking.planner.search import Frontend, trajectory_add_zero_velocity
from CarlaAutoParking.others.se2state import SE2State, generate_vehicle_vertexes
from CarlaAutoParking.others.geometry import (
    anticlockwise_vertexes_sort,
    point_distance_line,
    # line_parallel_line,
    line_line_angle,
)
from CarlaAutoParking.utils.plt_utils import image_list_to_video
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


class CameraPerception:
    def __init__(self, K, Z) -> None:
        self.K = K
        self.Z = Z
        self.mean_pix_thres = (190, 220)
        self.var_pix_thres = (0, 3500)

    def process_image(self, image):
        lines = extract_pixel_lines(image)

        parking_vertexes_list = []
        obstacle_vertexes_list = []

        for l1 in lines:
            l1_x1, l1_y1, l1_x2, l1_y2 = l1[0]
            a, b = (l1_x1, l1_y1), (l1_x2, l1_y2)

            for l2 in lines:
                l2_x1, l2_y1, l2_x2, l2_y2 = l2[0]
                c, d = (l2_x1, l2_y1), (l2_x2, l2_y2)
                if line_line_angle([a, b], [c, d]) > 0.2:
                    continue

                sorted_pixel_point_list = anticlockwise_vertexes_sort([a, b, c, d])
                sorted_physic_point_list = pixel_point_list_to_physics(
                    sorted_pixel_point_list, self.K, self.Z
                )
                if is_candidate_parking_rect(sorted_physic_point_list, self.K, self.Z):
                    mean_pix, var_pix, box = get_area_pixel(
                        image, sorted_pixel_point_list
                    )
                    if (
                        self.mean_pix_thres[0] < mean_pix < self.mean_pix_thres[1]
                        and var_pix < self.var_pix_thres[1]
                    ):
                        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                        parking_vertexes_list = sorted_physic_point_list
                        # cv2.imshow("image line", image)

                    else:
                        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
                        obstacle_vertexes_list = sorted_physic_point_list
                        # cv2.imshow("image line", image)

        cv2.imshow("image line", image)
        return parking_vertexes_list, obstacle_vertexes_list, image


def extract_pixel_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray, 50, 150, apertureSize=3
    )  # apertureSize是sobel算子大小，只能为1,3,5，7
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
    )

    return lines


def pixel_uv_to_physics_xyz(uv, inv_K, Z):
    p1 = np.array([uv[0], uv[1], 1])
    xyz1 = Z * inv_K @ p1

    return (xyz1[0], xyz1[1])


def pixel_point_list_to_physics(pixel_point_list, K, Z):
    inv_K = np.linalg.inv(K)

    physics_points = []
    for pixel_point in pixel_point_list:
        physics_points += [pixel_uv_to_physics_xyz(pixel_point, inv_K, Z)]

    return physics_points


def is_candidate_parking_rect(sorted_physic_point_list, K, Z):
    vec_ab = np.array(sorted_physic_point_list[1]) - np.array(
        sorted_physic_point_list[0]
    )
    vec_bc = np.array(sorted_physic_point_list[2]) - np.array(
        sorted_physic_point_list[1]
    )
    vec_cd = np.array(sorted_physic_point_list[3]) - np.array(
        sorted_physic_point_list[2]
    )
    vec_da = np.array(sorted_physic_point_list[0]) - np.array(
        sorted_physic_point_list[3]
    )

    if (abs(vec_ab @ vec_bc) > 0.3) or (abs(vec_cd @ vec_da) > 0.3):
        return False

    len_ab = np.linalg.norm(vec_ab)
    len_bc = np.linalg.norm(vec_bc)
    len_cd = np.linalg.norm(vec_cd)
    len_da = np.linalg.norm(vec_da)
    max_len = max(len_ab, len_bc, len_cd, len_da)
    min_len = min(len_ab, len_bc, len_cd, len_da)

    if float(max_len / min_len) < 1.5:
        return False

    area = get_rect_area(*sorted_physic_point_list)

    if area < 12 or area > 18:
        return False

    return True


def get_rect_area(p1, p2, p3, p4):
    def getS(a, b, c):
        """
        Get triangle area.
        """
        return abs(
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
        )

    total_area = 0
    total_area += getS(p1, p2, p3)
    total_area += getS(p1, p3, p4)

    return total_area


def get_area_pixel(image: np.ndarray, sorted_pixel_point_list):
    # 假设我们有两个对角点的坐标
    x1, y1 = sorted_pixel_point_list[0]
    x2, y2 = sorted_pixel_point_list[2]

    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    vec_ab = np.array(sorted_pixel_point_list[1]) - np.array(sorted_pixel_point_list[0])
    vec_bc = np.array(sorted_pixel_point_list[2]) - np.array(sorted_pixel_point_list[1])
    vec_cd = np.array(sorted_pixel_point_list[3]) - np.array(sorted_pixel_point_list[2])

    angle = np.arctan2(vec_ab[1], vec_ab[0]) * 180 / np.pi
    len_ab = np.linalg.norm(vec_ab)  # w
    len_bc = np.linalg.norm(vec_bc)  # h

    rect = ((center_x, center_y), (len_ab, len_bc), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 创建掩模
    mask = np.zeros((640, 640), np.uint8)

    # 在掩模上绘制旋转矩形的轮廓
    cv2.drawContours(mask, [box], 0, 255, -1)

    # 提取旋转矩形内的像素
    rotated_rect_pixels = cv2.bitwise_and(image, image, mask=mask)

    # 计算均值（所有通道）
    mean_all_channels = np.mean(rotated_rect_pixels[mask == 255], axis=(0, 1))

    # 计算方差（所有通道）
    var_all_channels = np.var(rotated_rect_pixels[mask == 255], axis=(0, 1))

    return mean_all_channels, var_all_channels, box


def get_camera_intrinsic_transform(fov_x=90, pixel_width=640, pixel_height=640):
    fov_y = pixel_height / pixel_width * fov_x

    camera_intrinsics = np.zeros((3, 4))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = (pixel_width / 2.0) / np.tan(np.deg2rad(fov_x / 2))
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = (pixel_height / 2.0) / np.tan(np.deg2rad(fov_y / 2))
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics


def test():
    bev_video_path = str(pathlib.Path(__file__).parent) + "/videos/bev.mp4"

    cap = cv2.VideoCapture(bev_video_path)

    K = get_camera_intrinsic_transform()
    K = K[:, :3]
    Z = 8
    parking_detection = CameraPerception(K, Z)
    image_list = []
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is not None:
            _, _, image = parking_detection.process_image(frame)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list += [rgb]
        else:
            image_list_to_video(
                image_list, "ParkingSimulation/videos/parking_slot_detection.mp4"
            )
            break

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # 释放，关闭
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
