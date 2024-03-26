import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, TypeVar
from CarlaAutoParking.others.geometry import Polygon
from CarlaAutoParking.others.se2state import SE2State, SE2

Point = TypeVar("Point")

import pickle
import time

import random

from functools import partial


class GUI:
    def __init__(self) -> None:
        self.figure_num = random.randint(80, 100)
        self.fig = plt.figure(num=self.figure_num, figsize=[8, 8])
        self.fig.canvas.mpl_connect("button_release_event", self.on_click)
        self.select_point = None

        self.polygon_list: List[Polygon] = []
        self.vehicle_list: List[SE2] = []
        self.se2state_list: List[SE2State] = []
        self.image_list: List[np.ndarray] = []

        plt.gca().invert_yaxis()  # y轴反向
        plt.plot(10, 10, "ro")
        plt.draw()
        plt.title("Auto Parking")
        plt.pause(0.01)

        # self.looptest()

    def looptest(self):
        for _ in range(250):
            plt.figure(num=self.figure_num, figsize=[8, 8])
            plt.draw()
            plt.pause(0.02)
            self.updata_image_list()
        self.close()

    def update_vehicle(self, se2state: SE2State, line_type="-g"):
        plt.sca(self.fig.gca())
        if se2state.se2 in self.vehicle_list:
            return
        else:
            self.vehicle_list += [se2state.se2]
            self.se2state_list += [se2state]
            plot_vehicle(se2state, line_type)
            plt.pause(0.01)

        self.updata_image_list()

    def update_path(self, path: List[SE2State], path_name="path"):
        plt.sca(self.fig.gca())
        plot_path(path, path_name)
        plt.draw()
        plt.pause(1)

        self.updata_image_list()

    def update_polygon(self, polygon_list: List[Polygon], line_type="-b"):
        if len(polygon_list) < 1:
            return
        plt.sca(self.fig.gca())
        for polygon in polygon_list:
            if polygon in self.polygon_list:
                pass
            else:
                self.polygon_list += [polygon]
                plot_polygon(polygon, linetype=line_type)
        plt.pause(0.01)
        self.updata_image_list()

    def updata_image_list(self):
        ncols, nrows = self.fig.canvas.get_width_height()
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            nrows, ncols, 3
        )
        self.image_list += [image]

    def on_click(self, event):
        # print(
        #     "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
        #     % (
        #         "double" if event.dblclick else "single",
        #         event.button,
        #         event.x,
        #         event.y,
        #         event.xdata,
        #         event.ydata,
        #     )
        # )

        if self.select_point is None:
            print(
                f"Select parking polygon. Point ({event.xdata:5.2f},{event.ydata:5.2f})"
            )
            plt.sca(self.fig.gca())
            plt.scatter(event.xdata, event.ydata, s=250, c="m", marker="*")
            plt.draw()
            self.select_point = (event.xdata, event.ydata)

    def close(self):
        if len(self.image_list) > 5:
            fps = 50
            dt = 0.2
            N = len(self.se2state_list)

            if N > 5:
                dt = self.se2state_list[-1].t - self.se2state_list[-2].t
                fps = int(1 / dt)

            for i in range(N):
                if abs(self.se2state_list[i].v) > 0.1:
                    index = max(i - 10, 0)

            folder = str(pathlib.Path(__file__).parent.parent) + "/videos/"
            file_name = folder + str(time.time_ns()) + ".mp4"
            image_list_to_video(self.image_list[index:], file_name, fps)


def save_pickle_file(variable, file_name=None):
    if file_name is None:
        file_name = time.asctime()

    with open(file_name, "wb") as f:
        pickle.dump(variable, f)


def load_pickle_file(file_name=None):
    with open(file_name, "rb") as f:
        variable = pickle.load(f)

    return variable


def plot_line(p1: Tuple, p2: Tuple, linetype="-b", alpha=1.0):
    x = np.hstack((p1[0], p2[0]))
    y = np.hstack((p1[1], p2[1]))
    plt.plot(x, y, linetype, alpha=alpha)
    plt.draw()


def plot_polygon(polygon: Polygon, linetype="-b", alpha=1.0):
    plot_polygon_vertexes(polygon.vertexes, linetype, alpha)


def plot_polygon_vertexes(vertexes_list: List[Point], linetype="-b", alpha=1.0):
    point_array = np.array(vertexes_list)
    point_array = np.vstack((point_array, point_array[0, :]))
    plt.plot(point_array[:, 0], point_array[:, 1], linetype, alpha)
    plt.draw()


def plot_vehicle(se2state: SE2State, line_type="-g"):
    from CarlaAutoParking.others.se2state import (
        generate_vehicle_vertexes,
        generate_wheels_vertexes,
    )

    v_b = generate_vehicle_vertexes(se2state)
    plot_polygon_vertexes(v_b, line_type, alpha=0.3)

    plt.plot(se2state.x, se2state.y, "gs", alpha=0.3)  # rear axle center.

    for v_w in generate_wheels_vertexes(se2state):
        plot_polygon_vertexes(v_w, "-r", alpha=0.2)

    arrow_length = np.clip(se2state.v, -2, 2)

    plt.arrow(
        se2state.x,
        se2state.y,
        arrow_length * np.cos(se2state.heading),
        arrow_length * np.sin(se2state.heading),
        head_width=0.35,
        fc="r",
        ec="g",
    )
    plt.draw()


def plot_task(
    obstacle_polygon_list: List[Polygon],
    start_se2state: SE2State,
    goal_se2state: SE2State,
):
    for obs_polygon in obstacle_polygon_list:
        plot_polygon(obs_polygon)

    plt.axis("equal")

    plot_vehicle(start_se2state)
    plt.text(
        start_se2state.x,
        start_se2state.y,
        f" {start_se2state.heading:5.2f}",
        color="b",
    )

    plot_vehicle(goal_se2state)
    plt.text(
        start_se2state.x,
        start_se2state.y,
        f" {start_se2state.heading:5.2f}",
        color="r",
        size=10,
    )
    plt.draw()


def plot_path(path: List[SE2State], label="path"):
    x = np.array([se2.x for se2 in path])
    y = np.array([se2.y for se2 in path])
    plt.plot(x, y, label=label)
    plt.legend()
    plt.draw()
    # plt.pause(0.001)


def plot_trajectory_animation(path: List[SE2State]):
    for se2 in path:
        plot_vehicle(se2)
        plt.draw()
        plt.pause(0.01)


def plot_control(path: List[SE2State]):
    v = np.array([se2.v for se2 in path])
    a = np.array([se2.a for se2 in path])
    j = np.array([se2.jerk for se2 in path])
    d = np.array([se2.delta for se2 in path])
    dd = np.array([se2.delta_dot for se2 in path])
    x = np.array([se2.x for se2 in path])
    y = np.array([se2.y for se2 in path])
    heading = np.array([se2.heading for se2 in path])

    N = 8
    len_path = len(path)
    if path[-1].t > 0:
        dt = abs(path[1].t - path[0].t)
        T = dt * (len_path - 1)
    else:
        T = 0.02 * (len_path - 1)

    t = np.linspace(0, T, len_path)
    plt.subplot(N, 1, 1)
    plt.plot(t, v, label="v")
    plt.legend()

    plt.subplot(N, 1, 2)
    plt.plot(t, a, label="a")
    plt.legend()

    plt.subplot(N, 1, 3)
    plt.plot(t, j, label="jerk")
    plt.legend()

    plt.subplot(N, 1, 4)
    plt.plot(t, d, label="delta")
    plt.legend()

    plt.subplot(N, 1, 5)
    plt.plot(t, dd, label="delta_dot")
    plt.legend()

    plt.subplot(N, 1, 6)
    plt.plot(t, x, label="x")
    plt.legend()
    plt.subplot(N, 1, 7)
    plt.plot(t, y, label="y")
    plt.legend()

    plt.subplot(N, 1, 8)
    plt.plot(t, heading, label="heading")
    plt.legend()


def plot_heatmap(data: np.ndarray):
    data = np.array(data).T
    plt.gcf()
    im = plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)
    plt.colorbar(im, extend="both", extendrect=True)
    plt.draw()


import imageio.v2 as imageio


def record_gif(file_name=None, image_list=None, duration=0.02):
    """
    Make a gif image for giving image folder
    """

    imageio.mimsave(file_name, image_list, "GIF", duration=duration)


import moviepy.editor as me
import moviepy
import cv2


def gif_to_video(gif_in_path, gif_out_path):
    clip = me.VideoFileClip(gif_in_path)
    clip.write_videofile(gif_out_path)


def video_to_ndarray_list(video_path):
    rgb_frame_list = []
    video_read_capture = cv2.VideoCapture(video_path)
    while video_read_capture.isOpened():
        result, frame = video_read_capture.read()
        if not result:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame_list.append(rgb_frame)

    video_read_capture.release()

    return rgb_frame_list


def ndarray_list_to_video(numpy_array_list, video_out_path, fps=50):
    video_height = numpy_array_list[0].shape[0]
    video_width = numpy_array_list[0].shape[1]

    out_video_size = (video_width, video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc(*"mp4v"))
    video_write_capture = cv2.VideoWriter(
        video_out_path, output_video_fourcc, fps, out_video_size
    )

    for frame in numpy_array_list:
        video_write_capture.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_write_capture.release()


def image_list_to_video(image_list, video_out_path, fps=50):
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_list, fps=fps)
    clip.write_videofile(video_out_path)


def make_cover(text="hello world", size=(640, 640), font_size=2):
    """
    size (h, w)
    """
    h = size[0]
    w = size[1]
    c = 3
    covor = np.zeros((h, w, c), dtype=np.uint8)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_size
    font_color = (255, 255, 255)
    font_thickness = 3

    text_list = text.split("\n")
    n_line = len(text_list)

    # text_size (w, h)
    text_size, _ = cv2.getTextSize(text_list[0], font_face, font_scale, font_thickness)
    line_height = text_size[1] + int(0.5 * text_size[1])
    y0 = int(0.5 * (h - n_line * line_height))
    for i, line in enumerate(text_list):
        text_size, _ = cv2.getTextSize(line, font_face, font_scale, font_thickness)
        x = int(w / 2 - text_size[0] / 2)
        y = y0 + i * line_height
        cv2.putText(
            covor, line, (x, y), font_face, font_scale, font_color, font_thickness, 8, 0
        )

    return covor


def video_add_cover(video_in_path, video_out_path, text="hello world", font_size=2):
    frame_list = video_frame_list_add_cover(video_in_path, text, font_size)
    image_list_to_video(frame_list, video_out_path)


def video_frame_list_add_cover(video_in_path, text="hello world", font_size=2):
    frame_list: List = video_to_ndarray_list(video_in_path)

    h, w, c = frame_list[0].shape
    cover_begin = make_cover(text, (h, w), font_size)
    cover_end = np.zeros((h, w, c), dtype=np.uint8)
    frame_list = [cover_begin] * 75 + frame_list + [cover_end] * 20

    return frame_list


from moviepy.editor import VideoFileClip, AudioFileClip, vfx, afx


def video_add_bgm(video_in_path, video_out_path, audio_in_path):
    video_clip = me.VideoFileClip(video_in_path)
    audio_clip = me.AudioFileClip(audio_in_path)

    # fadein_clip = audio_clip.fadein(duration=1)
    # fadeout_clip = fadein_clip.fadeout(duration=1)

    fadein_clip = audio_clip.fx(me.afx.audio_fadein, duration=1.0)

    fadein_clip = fadein_clip.set_end(video_clip.duration - 1)
    fadeout_clip = fadein_clip.fx(me.afx.audio_fadeout, duration=1.0)

    # fadeout_clip = fadein_clip.set_end(video_clip.duration - 1).fadeout(duration=1)

    final_clip = video_clip.set_audio(fadeout_clip)
    final_clip.write_videofile(video_out_path)


def test_make_video():
    video_input_file = "ParkingSimulation/videos/video1.mp4"

    frame_list_1 = video_frame_list_add_cover(video_input_file, "Case 1.")
    video_input_file = "ParkingSimulation/videos/video2.mp4"
    frame_list_2 = video_frame_list_add_cover(video_input_file, "Case 2.")
    video_input_file = "ParkingSimulation/videos/video3.mp4"
    frame_list_3 = video_frame_list_add_cover(video_input_file, "Case 3.")

    cover = make_cover("Auto Parking.", font_size=2.5)
    black_img = np.zeros((640, 640, 3), dtype=np.uint8)

    output_frame_list = (
        [cover] * 75 + [black_img] * 20 + frame_list_1 + frame_list_2 + frame_list_3
    )

    video_output_file = "ParkingSimulation/videos/video123.mp4"

    image_list_to_video(output_frame_list, video_output_file)


def test_make_cover():
    covor = make_cover("hello world")
    cv2.imshow("cover", covor)
    cv2.waitKey()


def test_gif():
    gif_path = "ParkingSimulation/videos/video11.mp4"
    gif_out_path = "ParkingSimulation/videos/video11.mp4"
    gif_to_video(gif_path, gif_out_path)


def test_make_video():
    video_input_file = "ParkingSimulation/videos/video1.mp4"

    frame_list_1 = video_frame_list_add_cover(video_input_file, "Case 1.")
    video_input_file = "ParkingSimulation/videos/video2.mp4"
    frame_list_2 = video_frame_list_add_cover(video_input_file, "Case 2.")
    video_input_file = "ParkingSimulation/videos/video3.mp4"
    frame_list_3 = video_frame_list_add_cover(video_input_file, "Case 3.")

    cover = make_cover("Auto Parking.", font_size=2.5)
    black_img = np.zeros((640, 640, 3), dtype=np.uint8)

    output_frame_list = (
        [cover] * 75 + [black_img] * 20 + frame_list_1 + frame_list_2 + frame_list_3
    )

    video_output_file = "ParkingSimulation/videos/video123.mp4"

    image_list_to_video(output_frame_list, video_output_file)


def make_video():
    video_input_file = "ParkingSimulation/videos/parking_slot_detection.mp4"

    frame_list_1 = video_to_ndarray_list(video_input_file)

    cover = make_cover("Parking slot\nDetection.", font_size=2.5)
    black_img = np.zeros((640, 640, 3), dtype=np.uint8)

    output_frame_list = (
        [cover] * 75 + [black_img] * 20 + frame_list_1 + [black_img] * 10
    )

    video_output_file = "ParkingSimulation/videos/parking_slot_detection1.mp4"

    image_list_to_video(output_frame_list, video_output_file)
    pass


def add_bgm():
    video_input_file = "ParkingSimulation/videos/parking_slot_detection1.mp4"
    video_output_file = "ParkingSimulation/videos/parking_slot_detection11.mp4"
    audio_input_file = "ParkingSimulation/videos/OverTime.mp3"
    video_add_bgm(video_input_file, video_output_file, audio_input_file)


if __name__ == "__main__":
    # make_video()
    add_bgm()

    # test_make_cover()
    # test_video()
    # test_gif()
    # gui = GUI()
    # gui.looptest()
