import numpy as np
from typing import Tuple, TypeVar, Optional, List, Tuple
from copy import deepcopy

"""
Geometry functions.
"""

Point = TypeVar("Point", Tuple, List)

Line = TypeVar("Line", List[Tuple], List[List])

Circle = TypeVar("Circle", Tuple, List)  # x, y ,radius


class Polygon:
    """
    Convex Polygon, anticlockwise vertexes/lines.
    """

    def __init__(self, vertexes: List[Point]):
        # anticlockwise_vertexes_sort(vertexes)
        self.vertexes: List[Point] = anticlockwise_vertexes_sort(vertexes)
        self.lines: List[Line] = vertexes_to_lines(self.vertexes)
        self.norms: List[Point] = lines_to_norm(self.lines)
        self.center: Point = tuple(np.mean(np.array(self.vertexes), axis=0))

    @property
    def ndarray(self):
        return np.array(self.vertexes).T

    def __eq__(self, other: object) -> bool:
        if type(other).__name__ == "Polygon":
            if len(self.vertexes) != len(other.vertexes):
                return False
            array1 = np.array(self.vertexes)
            array2 = np.array(other.vertexes)

            if np.max(abs(array1 - array2)) > 1e-2:
                return False

            return True

        else:
            raise TypeError("Unsupported operand type for ==")

    def __repr__(self) -> str:
        return "polygon vertexes: " + str(self.vertexes)


class PolygonContainer:
    def __init__(self) -> None:
        self.polygon_list = []
        self.N = 0
        self.iter_index = 0

    def __len__(self):
        return len(self.polygon_list)

    def __getitem__(self, index):
        if index > self.N:
            raise IndexError("Index out range.")

        return self.polygon_list[index]

    def __setitem__(self, index, value: Polygon):
        if index > self.N:
            raise IndexError("Index out range.")

        if not type(value).__name__ == "Polygon":
            raise TypeError("Unsupported operand type for []")

        self.polygon_list[index] = value

    def __add__(self, other: Polygon) -> bool:
        if type(other).__name__ == "Polygon":
            if other in self.polygon_list:
                pass
            else:
                self.polygon_list += [other]

            return PolygonContainer(self.polygon_list)
        else:
            raise TypeError("Unsupported operand type for +")

    def __iadd__(self, other: Polygon) -> bool:
        if type(other).__name__ == "Polygon":
            if other in self.polygon_list:
                pass
            else:
                self.polygon_list += [other]
                self.N += 1

            return self
        else:
            raise TypeError("Unsupported operand type for +=")

    def __sub__(self, other):
        if type(other).__name__ == "Polygon":
            if other in self.polygon_list:
                self.polygon_list.remove(other)
            else:
                pass
            return PolygonContainer(self.polygon_list)
        else:
            raise TypeError("Unsupported operand type for -")

    def __isub__(self, other):
        if type(other).__name__ == "Polygon":
            if other in self.polygon_list:
                self.polygon_list.remove(other)
                self.N -= 1
            else:
                pass

            return self
        else:
            raise TypeError("Unsupported operand type for -=")

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        last_index = self.iter_index
        if last_index >= len(self.polygon_list):
            raise StopIteration
        else:
            self.iter_index += 1
            return self.polygon_list[last_index]


def ndarray_to_vertexlist(vertexs_array: np.ndarray):
    """
    vertexs_array: 2 * n, n * 2
    """

    nx, n = vertexs_array.shape
    if nx != 2 and n == 2:
        tmp = nx
        nx = n
        n = tmp
        vertexs_array = vertexs_array.T
    elif nx == 2 and n != 2:
        pass
    else:
        raise ValueError("Check numpy array shape!")

    vertexlist = []
    for i in range(n):
        vertexlist += [(vertexs_array[0, i], vertexs_array[1, i])]

    return vertexlist


def move_vertexes_array(
    vertexs_array: np.ndarray, rot_angle: float, offset: np.ndarray
):
    """
    ### move vertexs, coord is fixed, change points.
    rot_angle [rad].
    """
    nv, n = vertexs_array.shape
    no, n = offset.shape
    if nv != 2 or no != 2:
        raise ValueError("Check numpy array shape! 2 * n")

    rot = np.array(
        [
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )
    offset = np.array(offset).reshape((2, 1))
    return rot @ vertexs_array + offset


def change_vertexes_array_coord(
    vertexs_array: np.ndarray, rot_angle: float, offset: np.ndarray
):
    """
    ### change vertexs coord, point is fixed, change coord.
    ---
    rot_angle [rad]. rotate current coord to target coord

    offset [m]. trans current coord to target coord
    """
    nv, n = vertexs_array.shape
    no, n = offset.shape
    if nv != 2 or no != 2:
        raise ValueError("Check numpy array shape! 2 * n")

    rot = np.array(
        [
            [np.cos(rot_angle), np.sin(rot_angle)],
            [-np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )

    return rot @ (vertexs_array - offset)


def to_left(line: Line, point: Point):
    """
    ### 2D To left test.

    l: line [(x1, y1), (x2, y2)]
    p: point (x1, y1)
    """
    vec1 = np.array(line[1]) - np.array(line[0])
    vec2 = np.array(point) - np.array(line[0])
    return np.cross(vec1, vec2) > 0


# def anticlockwise_vertexes_sort(vertexes: List[Point]):
#     """
#     ### anticlockwise sort.
#     """

#     vertexes_array = np.array(vertexes).T  # 2 * N
#     center_x, center_y = np.mean(vertexes_array, axis=1)

#     n = len(vertexes)

#     for i in range(n):
#         for j in range(n - i - 1):
#             line = [(center_x, center_y), (vertexes[j][0], vertexes[j][1])]
#             point = (vertexes[j + 1][0], vertexes[j + 1][1])
#             if not to_left(line, point):
#                 temp = vertexes[j]
#                 vertexes[j] = vertexes[j + 1]
#                 vertexes[j + 1] = temp

#     sorted_vertexes = vertexes
#     return sorted_vertexes


def get_bottom_point(vertexes: List[Point]):
    min_index = 0
    n = len(vertexes)
    for i in range(n):
        if vertexes[i][1] < vertexes[min_index][1] or (
            vertexes[i][1] == vertexes[min_index][1]
            and vertexes[i][0] < vertexes[min_index][0]
        ):
            min_index = i
    return min_index


def pointset_to_convex_hull(vertexes_list: List[Point]):
    N = len(vertexes_list)
    sorted_vertexes = anticlockwise_vertexes_sort(vertexes_list)

    if N < 3:
        raise ValueError("point too small.")
    if N == 3:
        return sorted_vertexes

    from scipy.spatial import ConvexHull

    hull = ConvexHull(np.array(sorted_vertexes))
    hull_array = hull.points[hull.vertices, :].T
    return ndarray_to_vertexlist(hull_array)


def anticlockwise_vertexes_sort(vertexes: List[Point]):
    """
    ### anticlockwise sort.
    """

    vertexes_array = np.array(vertexes).T
    center_x, center_y = np.mean(vertexes_array, axis=1)
    point_with_angle = []
    n = len(vertexes)
    for i in range(n):
        atan2 = np.arctan2(vertexes[i][1] - center_y, vertexes[i][0] - center_x)
        if atan2 < 0:
            atan2 += 2 * np.pi
        point_with_angle += [(vertexes[i], atan2)]

    for i in range(n):
        for j in range(n - i - 1):
            if point_with_angle[j][1] > point_with_angle[j + 1][1]:
                temp = point_with_angle[j]
                point_with_angle[j] = point_with_angle[j + 1]
                point_with_angle[j + 1] = temp

    sorted_vertexes = [vertex for vertex, _ in point_with_angle]

    return sorted_vertexes


def line_intersect_line(l1: Line, l2: Line):
    """
    ### Line interset line test.

    point: (x, y)
    l1: [point, point]
    l2: [point, point]
    """
    if to_left(l2, l1[0]) ^ to_left(l2, l1[1]):  # 异或， 一个在左边，一个在右边
        if to_left(l1, l2[0]) ^ to_left(l1, l2[1]):
            return True

    return False


def point_in_circle(point: Point, circle: Circle):
    """
    ### Point in circle test.

    circle: (x, y, r)
    point: (x, y)

    """
    if np.hypot(point[0] - circle[0], point[1] - circle[1]) < circle[2]:
        return True
    return False


def line_intersect_circle(line: Line, circle):
    """
    ### Line intersect circle test.
    circle: (x, y, r)
    line: [p1, p2]
    """
    if point_in_circle(line[0], circle) or point_in_circle(line[1], circle):
        return True

    oa = np.array([circle[0] - line[0][0], circle[1] - line[0][1]])
    ob = np.array([circle[0] - line[1][0], circle[1] - line[1][1]])
    ao = -oa
    bo = -ob
    ab = np.array([line[0][0] - line[1][0], line[0][1] - line[1][1]])
    ba = -ab
    d = abs(np.cross(ab, ob) / np.linalg.norm(ab))

    if d <= circle[2]:
        if np.dot(ao, ab) > 0 and np.dot(bo, ba) > 0:
            return True


def vertexes_to_lines(vertexes: List[Point]):
    """
    ### From anticlockwise vertexes get anticlockwise lines.
    """
    lines = []
    newvertexes = deepcopy(vertexes)
    newvertexes.append(newvertexes[0])
    for i in range(len(newvertexes) - 1):
        lines.append((newvertexes[i], newvertexes[i + 1]))

    return lines


def lines_to_norm(lines: List[Line]):
    """
    ### Return every norm vector without normlize.
    """
    norms = []

    for line in lines:
        vec = np.array(line[0]) - np.array(line[1])
        norms.append((vec[1], -vec[0]))  # (y,-x) in the left

    return norms


def vertexes_to_norm(vertexes: List[Point]):
    lines = vertexes_to_lines(vertexes)
    return lines_to_norm(lines)


def get_polygon_area(polygon: Polygon):
    def getS(a, b, c):
        """
        Get triangle area.
        """
        return abs(
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
        )

    total_area = 0
    for i in range(1, len(polygon.vertexes) - 1):
        total_area += getS(
            polygon.vertexes[0], polygon.vertexes[i], polygon.vertexes[i + 1]
        )

    return total_area


def polygon_intersect_polygon(polygon1: Polygon, polygon2: Polygon):
    def dotProduct(nor, points: list):
        res = []

        for p in points:
            res.append(nor[0] * p[0] + nor[1] * p[1])
        return (min(res), max(res))

    sep_axis = polygon1.norms + polygon2.norms

    for sep in sep_axis:
        res1 = dotProduct(sep, polygon1.vertexes)
        res2 = dotProduct(sep, polygon2.vertexes)

        if res1[1] < res2[0] or res1[0] > res2[1]:
            return False
        else:
            continue
    return True


def polygon_intersect_line(polygon: Polygon, line: Line):
    """
    ### Line intersect this polygon ?
    line: [p1(x,y), p2]
    """

    for l in polygon.lines:
        if line_intersect_line(line, l):
            return True

    return False


# def point_in_polygon(point: Point, polygon: Polygon):
#     """
#     ### Point in polygon ?

#     Point: (x, y)
#     """

#     for l in polygon.lines:
#         if not to_left(l, point):
#             return False

#     return True


def point_in_polygon(point: Point, polygon: Polygon):
    """
    ### Point in polygon ?

    Point: (x, y)
    """

    px, py = point
    polygon_array = np.array(polygon.vertexes)
    xmin, ymin = np.min(polygon_array, axis=0)
    xmax, ymax = np.max(polygon_array, axis=0)

    if px > xmax or px < xmin or py > ymax or py < ymin:
        return False

    N = len(polygon.vertexes)
    is_in = False

    for i, corner in enumerate(polygon.vertexes):
        next_i = i + 1 if i + 1 < N else 0
        x1, y1 = corner
        x2, y2 = polygon.vertexes[next_i]

        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break

        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in

        if py == y1 and py == y2:
            if min(x1, x2) < px < max(x1, x2):
                is_in = True
                break

    return is_in


def polygon_in_polygon(lhs_polygon: Polygon, rhs_polygon: Polygon):
    """
    ### Polygon in polygon ?

    """
    for vertex in lhs_polygon.vertexes:
        if not point_in_polygon(rhs_polygon, vertex):
            return False

    return True


def point_distance_line(point: Point, line: Line):
    """
    ### Point distance linesegment.

    """
    A, B = line[0], line[1]
    vec_AB = np.array(B) - np.array(A)
    vec_AP = np.array(point) - np.array(A)
    t = (vec_AB @ vec_AP) / (vec_AB @ vec_AB)

    if t >= 1:
        Dx = B[0]
        Dy = B[1]
    elif t > 0:
        Dx = A[0] + vec_AB[0] * t
        Dy = A[1] + vec_AB[1] * t
    else:
        Dx = A[0]
        Dy = A[1]

    vec_PD = [Dx - point[0], Dy - point[1]]
    return np.sqrt(vec_PD[0] * vec_PD[0] + vec_PD[1] * vec_PD[1])


def line_distance_line(line1: Line, line2: Line):
    a = line1[0]
    b = line1[1]
    c = line2[0]
    d = line2[1]

    dist_a_to_cd = point_distance_line(a, line2)
    dist_b_to_cd = point_distance_line(b, line2)
    dist_c_to_ab = point_distance_line(c, line1)
    dist_d_to_ab = point_distance_line(d, line1)

    return min(dist_a_to_cd, dist_b_to_cd, dist_c_to_ab, dist_d_to_ab)


def line_line_angle(line1: Line, line2: Line):
    line_vec1 = np.array([line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]])
    line_vec2 = np.array([line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]])

    theta_1 = np.arctan2(line_vec1[1], line_vec1[0])
    theta_2 = np.arctan2(line_vec2[1], line_vec2[0])

    if theta_1 * theta_2 >= 0:
        insideAngle = abs(theta_1 - theta_2)
    else:
        insideAngle = abs(theta_1) + abs(theta_2)
        if insideAngle > np.pi:
            insideAngle = 2 * np.pi - insideAngle
    return insideAngle


def point_distance_polygon(point: Point, polygon: Polygon):
    """
    ### Point distance polygon

    """
    dis_list = []

    for line in polygon.lines:
        dis = point_distance_line(point, line)
        dis_list += [dis]

    return min(dis_list)


def get_polygon_halfspaces(polygon: Polygon):
    """
    Return A, b, the polygon can represent A@[x,y] <= b
    [x,y] in polygon.
    """

    N = len(polygon.lines)
    A_ret = np.zeros((N, 2))
    b_ret = np.zeros((N, 1))
    for i in range(N):
        v1, v2 = polygon.lines[i][1], polygon.lines[i][0]
        ab = np.zeros((2, 1))

        if abs(v1[0] - v2[0]) < 1e-10:
            if v2[1] < v1[1]:
                Atmp = np.array([1, 0])
                btmp = v1[0]
            else:
                Atmp = np.array([-1, 0])
                btmp = -v1[0]
        elif abs(v1[1] - v2[1]) < 1e-10:
            if v1[0] < v2[0]:
                Atmp = np.array([0, 1])
                btmp = v1[1]
            else:
                Atmp = np.array([0, -1])
                btmp = -v1[1]
        else:
            temp1 = np.array([[v1[0], 1], [v2[0], 1]])
            temp2 = np.array([[v1[1]], [v2[1]]])
            ab = np.linalg.inv(temp1) @ temp2

            a = ab[0, 0]
            b = ab[1, 0]
            if v1[0] < v2[0]:
                Atmp = np.array([-a, 1])
                btmp = b
            else:
                Atmp = np.array([a, -1])
                btmp = -b

        A_ret[i, :] = Atmp
        b_ret[i, :] = btmp

    return A_ret, b_ret


def test_scale_halfspaces():
    import random

    shape_points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(3)]

    polygon = Polygon(shape_points)

    A, b = get_polygon_halfspaces(polygon)


def test_halfspaces():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    shape_points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(3)]

    polygon = Polygon(shape_points)

    A, b = get_polygon_halfspaces(polygon)

    for _ in range(100):
        point = (random.randint(0, 10), random.randint(0, 10))

        point_array = np.array([point])
        flag1 = point_in_polygon(polygon, point)
        flag2 = np.all(A @ point_array.T < b)

        plt.cla()
        plt_utils.plot_polygon(polygon)
        plt.plot(point[0], point[1], "ro")
        plt.draw()
        plt.pause(0.1)
        if flag1 == flag2:
            print(f"\033[032m[test halfspaces pass, {flag1}]\033[0m")
        else:
            print(f"\033[031m[test halfspaces fail, {flag1}]\033[0m")


def test_single_area():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    shape_point = [(9.75, 3.0), (7.25, 3.0), (7.25, 9.0), (9.75, 9.0)]

    vehicle_point = (9.4555, 5.60)

    obstacle_polygon = Polygon(shape_point)
    area = get_polygon_area(obstacle_polygon)
    plt_utils.plot_polygon(obstacle_polygon)
    plt.plot(vehicle_point[0], vehicle_point[1], "ro")
    plt.draw()
    plt.pause(0.1)
    total_area = 0
    for l in obstacle_polygon.lines:
        a, b, c = l[0], l[1], vehicle_point
        total_area += (
            np.fabs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
        )
    print(f"\033[032m[in polygon , {area,total_area}]\033[0m")
    plt.show()


def test_area():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    for i in range(100):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(4)
        ]
        convexhull_points = pointset_to_convex_hull(shape_points)
        convex_polygon = Polygon(convexhull_points)
        plt_utils.plot_polygon(convex_polygon)

        plt.draw()
        area = get_polygon_area(convex_polygon)

        point = (random.randint(0, 10), random.randint(0, 10))
        plt.plot(point[0], point[1], "ro")
        plt.draw()
        plt.pause(0.5)

        total_area = 0
        for l in convex_polygon.lines:
            a, b, c = l[0], l[1], point
            # total_area += (
            #     np.fabs(
            #         c[0] * a[1]+ a[0] * b[1]+ b[0] * c[1]- c[0] * b[1]- a[0] * c[1]- b[0] * a[1]
            #     )
            #     * 0.5
            # )
            total_area += (
                np.fabs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                * 0.5
            )
        if point_in_polygon(convex_polygon, point):
            if abs(total_area - area) < 1e-3:
                print(f"\033[032m[in polygon , test pass, {area==total_area}]\033[0m")
            else:
                print(f"\033[031m[in polygon , test fail, {area,total_area}]\033[0m")
        else:
            if abs(total_area - area) < -1e-3:
                print(f"\033[031m[out polygon , test fail, {area,total_area}]\033[0m")
            else:
                print(f"\033[032m[out polygon , test pass, {total_area>=area}]\033[0m")

        plt.pause(0.1)
        plt.cla()


def test_convex_hull():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    for i in range(50):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        ]
        plt.cla()
        plt_utils.plot_polygon(Polygon(shape_points))
        plt.pause(0.1)
        plt.draw()
        convexhull_points = pointset_to_convex_hull(shape_points)
        plt_utils.plot_polygon(Polygon(convexhull_points))
        plt.pause(0.5)
        plt.draw()


def test_polygon_eq():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    for i in range(50):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        ]

        p1 = Polygon(shape_points)
        # shape_points = [
        #     (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        # ]
        p2 = Polygon(shape_points)
        if p1 == p2:
            print("p1 == p2")
        print(f"\033[032m[polygon_eq, test pass, {p1!=p2}]\033[0m")


def test_polygon_eq_list():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    for i in range(5):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        ]

        plist = [Polygon(shape_points)]

        p = Polygon(shape_points)
        if p in plist:
            print(f"\033[032m[polygon_eq, test pass, {p}]\033[0m")


def test_polygon_container():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    polygon_container1 = PolygonContainer()

    for i in range(5):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(7)
        ]

        polygon_container1 += Polygon(shape_points)

    for polygon in polygon_container1:
        print(polygon)

    print(polygon_container1.iter_index)
    print(len(polygon_container1))

    for polygon in polygon_container1:
        print(polygon)


def test_point_polygon():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    plt.figure()
    for i in range(50):
        shape_points = [
            (random.randint(0, 10), random.randint(0, 10)) for i in range(5)
        ]
        point = (random.randint(0, 10), random.randint(0, 10))
        polygon = Polygon(shape_points)

        plt.cla()
        plt_utils.plot_polygon(polygon)
        plt.plot(point[0], point[1], "ro")
        plt.pause(0.5)
        plt.draw()

        if point_in_polygon(point, polygon):
            distance = point_distance_polygon(point, polygon)
            print(f"point in polygon . min distance is {distance}")
            plt.pause(1)


def test_point_polygon1():
    import random
    import matplotlib.pyplot as plt
    import CarlaAutoParking.utils.plt_utils as plt_utils

    shape_points = [(0, 10), (0, 7), (5, 1)]
    shape_points = [(0, 5), (3, 3), (8, 6), (1, 9), (3, 1)]
    point = (0, 9)
    point = (2, 6)
    polygon = Polygon(shape_points)

    plt.cla()
    plt_utils.plot_polygon(polygon)
    plt.plot(point[0], point[1], "ro")
    plt.draw()
    plt.pause(0.1)

    # pointset_to_convex_hull()
    import time

    start = time.time_ns()

    for _ in range(10000):
        point_in_polygon(point, polygon)

    print(f"point in polygon time is {time.time_ns() - start:5d}")

    # print(f"is in{point_in_polygon(point, polygon)}")
    # print(f"distance {point_distance_polygon(point, polygon)}")


if __name__ == "__main__":
    test_point_polygon()
