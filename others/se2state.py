from config import VehicleConfig, GridMapConfig
from copy import deepcopy
import numpy as np
import CarlaAutoParking.others.geometry as geometry


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class SO2:
    def __init__(
        self,
        angle: float = None,
        a: float = None,
        b: float = None,
    ):
        if angle is None and a is not None and b is not None:
            sqrt = np.sqrt(a * a + b * b)
            self.a = a / sqrt
            self.b = b / sqrt
            self.heading = np.arctan2(b, a)
        elif angle is not None and a is None and b is None:
            self.heading = float(normalize_angle(angle))
            self.a = np.cos(self.heading)  # real
            self.b = np.sin(self.heading)  # image
            self.rotation = np.array(
                [[self.a, -self.b], [self.b, self.a]]
            )  # rotation matrixs
        else:
            raise ValueError("Unsupported initialize for SO2, check input.")

    @classmethod
    def from_complex(cls, a, b):
        return cls(None, a, b)

    @classmethod
    def from_angle(cls, angle):
        return cls(angle, None, None)

    def __add__(self, other):
        """
        rotate complex by other, anticlockwise "+"
        """

        if type(other).__name__ == "SO2":
            a = self.a * other.a - self.b * other.b
            b = self.b * other.a + self.a * other.b
            return SO2.from_complex(a=a, b=b)
        else:
            raise TypeError("Unsupported operand type for +")

    def __iadd__(self, other):
        if type(other).__name__ == "SO2":
            new_so = self.__add__(other)
            self.a = new_so.a
            self.b = new_so.b
            self.heading = new_so.heading
            self.rotation = new_so.rotation
            return self
        else:
            raise TypeError("Unsupported operand type for +=")

    def __sub__(self, other):
        """
        (q^{*})_current * q_other, other in current rotation., anticlockwise "+"
        """
        if type(other).__name__ == "SO2":
            square = other.a * other.a + other.b * other.b
            a = (self.a * other.a + self.b * other.b) / square
            # b = (self.b * other.a - self.a * other.b) / square
            b = (self.a * other.b - self.b * other.a) / square
            return SO2.from_complex(a=a, b=b)
        else:
            raise TypeError("Unsupported operand type for -")

    def __isub__(self, other):
        if type(other).__name__ == "SO2":
            new_so = self.__add__(other)
            self.a = new_so.a
            self.b = new_so.b
            self.heading = new_so.heading
            self.rotation = new_so.rotation
            return self
        else:
            raise TypeError("Unsupported operand type for -")

    def __eq__(self, other):
        if type(other).__name__ == "SO2":
            return abs(self.a - other.a) < 1e-4 and abs(self.b - other.b) < 1e-4
        else:
            raise TypeError("Unsupported operand type for ==")

    def __repr__(self):
        return f"(a:{self.a:4.2f}+ b:{self.b:4.2f}i), heading:{self.heading:4.2f} [rad]"


class SE2:
    def __init__(
        self,
        x: float = None,
        y: float = None,
        heading: float = None,
        so2: SO2 = None,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        if heading is not None and so2 is None:
            self.so2 = SO2.from_angle(heading)
        elif heading is None and so2 is not None:
            self.so2 = so2
        else:
            raise ValueError("Unsupported initialize for SE2")

    @classmethod
    def from_SO2(cls, x, y, so2: SO2):
        return cls(x, y, heading=None, so2=so2)

    @classmethod
    def from_heading(cls, x, y, heading: float):
        return cls(x, y, heading=heading, so2=None)

    def __add__(self, other):
        if type(other).__name__ == "SE2":
            x = self.x + other.x
            y = self.y + other.y
            so2 = self.so2 + other.so2

            return SE2.from_SO2(x, y, so2)
        else:
            raise TypeError("Unsupported operand type for +")

    def __iadd__(self, other):
        if type(other).__name__ == "SE2":
            new_se2 = self.__add__(other)
            self.x = new_se2.x
            self.y = new_se2.y
            self.so2 = new_se2.so2
            return self
        else:
            raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        if type(other).__name__ == "SE2":
            """
            Other in current vector.
            """
            x = -self.x + other.x
            y = -self.y + other.y
            so2 = self.so2 - other.so2
            return SE2.from_SO2(x, y, so2)
        else:
            raise TypeError("Unsupported operand type for -")

    def __isub__(self, other):
        if type(other).__name__ == "SE2":
            new_se2 = self.__sub__(other)
            self.x = new_se2.x
            self.y = new_se2.y
            self.so2 = new_se2.so2
            return self
        else:
            raise TypeError("Unsupported operand type for +")

    def __eq__(self, other):
        if type(other).__name__ == "SE2":
            return (
                abs(self.x - other.x) < 1e-4
                and abs(self.y - other.y) < 1e-4
                and self.so2 == other.so2
            )
        else:
            raise TypeError("Unsupported operand type for ==")

    def __repr__(self):
        return (
            f"(x:{self.x:4.2f}, y:{self.y:4.2f}, heading:{self.so2.heading:4.2f} [rad])"
        )

    def norm(self):
        return np.linalg.norm(self.x**2 + self.y**2 + self.so2.heading**2)


class SE2State:
    move_distance = 0.2

    def __init__(
        self,
        x: float = None,
        y: float = None,
        heading: float = None,
        se2: SE2 = None,
        gridmap_cfg=GridMapConfig(),
    ):
        # self.index = -1
        # self.visited = False
        if (se2 is not None) and (x is None and y is None and heading is None):
            self._se2 = se2

        elif (se2 is None) and (
            x is not None and y is not None and heading is not None
        ):
            self._se2 = SE2.from_heading(x, y, heading)
        else:
            raise ValueError("Unsupported initialize for SE2state, check input.")

        self.x_index = -1
        self.y_index = -1
        self.heading_index = -1
        self.direction_index = -1

        self.parent = None
        self.cost_to_here = 999999
        self.cost_to_goal = 999999

        self.t = 0.0
        self.v = 0.0
        self.a = 0.0
        self.jerk = 0.0
        self.delta = 0.0
        self.delta_dot = 0.0
        self.curv = 0.0

    @classmethod
    def from_se2(cls, se2: SE2):
        return cls(None, None, None, se2)

    @classmethod
    def from_xyh(cls, x: float, y: float, h: float):
        return cls(x, y, h, None)

    @property
    def x(self):
        return self._se2.x

    @property
    def y(self):
        return self._se2.y

    @property
    def heading(self):
        return self._se2.so2.heading

    @property
    def so2(self):
        return self._se2.so2

    @property
    def se2(self):
        return self._se2

    @se2.setter
    def se2(self, se2):
        self._se2 = se2

    @property
    def array_state(self):
        return np.array([self.x, self.y, self.heading, self.v, 0], dtype=np.float32)

    @property
    def array_input(self):
        return np.array([self.a, self.delta], dtype=np.float32)

    def __lt__(self, other):
        if type(other).__name__ == "SE2State":
            return self.cost() < other.cost()
        else:
            raise TypeError("Unsupported operand type for <")

    def __eq__(self, other) -> bool:
        if type(other).__name__ == "SE2State":
            return (
                self.x_index == other.x_index
                and self.y_index == other.y_index
                and self.heading_index == other.heading_index
            )
        else:
            raise TypeError("Unsupported operand type for ==")

    def __repr__(self):
        return f"(x:{self.x:4.2f}, y:{self.y:4.2f}, heading:{self.so2.heading:4.2f} [rad]), (v:{self.v:4.2f}, a:{self.a:4.2f}, delta:{self.delta:4.2f} [rad])"

    def get_index_3d(self):
        return self.x_index, self.y_index, self.heading_index

    def get_index_2d(self):
        return self.x_index, self.y_index

    def cost_to_state(self, state, vehicle_cfg=VehicleConfig()):
        # """
        # Optimal control cost, only terminal cost.
        # """

        # ddelta = state.delta / vehicle_cfg.max_front_wheel_angle * vehicle_cfg.max_v
        # stage_cost = (ddelta * ddelta + state.v * state.v) * vehicle_cfg.T

        """
        Path length cost.
        """
        heading = deepcopy(self.heading)
        path_len_cost = 0
        delta_t = 0.02
        discrete_num = int(vehicle_cfg.T / delta_t)

        for _ in range(discrete_num):
            dx = state.v * np.cos(heading) * delta_t
            dy = state.v * np.sin(heading) * delta_t
            heading += state.v / vehicle_cfg.wheel_base * np.tan(state.delta) * delta_t
            path_len_cost += np.sqrt(dx * dx + dy * dy)

        # next_v = deepcopy(state.v)
        # for _ in range(discrete_num):
        #     dv = state.a * vehicle_cfg.T

        #     next_v = next_v + dv
        #     heading += (
        #         next_v / vehicle_cfg.wheel_base * np.tan(state.delta) * vehicle_cfg.T
        #     )
        #     dx = next_v * np.cos(heading) * vehicle_cfg.T
        #     dy = next_v * np.sin(heading) * vehicle_cfg.T

        #     path_len_cost += np.sqrt(dx * dx + dy * dy)

        return path_len_cost

    def cost_to_gridstate(self, state):
        """
        Euclidian cost.
        """
        dx = state.x - self.x
        dy = state.y - self.y
        euclidian_cost = np.sqrt(dx * dx + dy * dy)
        return euclidian_cost

    def cost(self):
        return self.cost_to_here + self.cost_to_goal

    def is_close_enough(self, other):
        if type(other).__name__ == "SE2State":
            dx = self.x - other.x
            dy = self.y - other.y
            d = np.sqrt(dx * dx + dy * dy)
            dheading = abs((self.so2 - other.so2).heading)

            distance_flag = False
            if d < 0.1 and dheading < 0.1:
                distance_flag = True

            dv = abs(self.v - other.v)
            ddelta = abs(self.delta - other.delta)

            dynamic_flag = False

            if dv < 0.1 and ddelta < 0.1:
                dynamic_flag = True

            if distance_flag:
                return True
            else:
                return False

        else:
            raise TypeError("Unsupported for se2 is_close_enough().")


def generate_vehicle_vertexes(state: SE2State, vehicle_cfg=VehicleConfig()):
    """
    Default is anticlockwise order.
    """
    x = state.x
    y = state.y
    heading = state.heading
    b_to_f = vehicle_cfg.baselink_to_front
    b_to_r = vehicle_cfg.baselink_to_rear

    W = vehicle_cfg.width

    vertexs = np.array(
        [[-b_to_r, -W / 2], [b_to_f, -W / 2], [b_to_f, W / 2], [-b_to_r, W / 2]]
    ).T

    vertexs_body = geometry.move_vertexes_array(vertexs, heading, np.array([[x], [y]]))
    Vbody = geometry.ndarray_to_vertexlist(vertexs_body)

    return Vbody


def generate_wheels_vertexes(state: SE2State, vehicle_cfg=VehicleConfig()):
    """
    Default is anticlockwise order.
    """
    x = state.x
    y = state.y
    heading = state.heading

    delta = state.delta
    wheel_radius = vehicle_cfg.wheel_radius
    wheel_wdith = vehicle_cfg.wheel_width
    wheel_distance = vehicle_cfg.wheel_distance

    vertexs_wheel = np.array(
        [
            [-wheel_radius, -wheel_wdith],
            [wheel_radius, -wheel_wdith],
            [wheel_radius, wheel_wdith],
            [-wheel_radius, wheel_wdith],
        ]
    ).T

    wheel_base = vehicle_cfg.wheel_base

    rr_offset = np.array([[0], [-wheel_distance / 2]])
    fr_offset = np.array([[wheel_base], [-wheel_distance / 2]])
    fl_offset = np.array([[wheel_base], [wheel_distance / 2]])
    rl_offset = np.array([[0], [wheel_distance / 2]])

    rr_wheel = geometry.move_vertexes_array(vertexs_wheel, 0.0, rr_offset)
    fr_wheel = geometry.move_vertexes_array(vertexs_wheel, delta, fr_offset)
    fl_wheel = geometry.move_vertexes_array(vertexs_wheel, delta, fl_offset)
    rl_wheel = geometry.move_vertexes_array(vertexs_wheel, 0, rl_offset)

    xy_offset = np.array([[x], [y]])
    rr_wheel = geometry.move_vertexes_array(rr_wheel, heading, xy_offset)
    fr_wheel = geometry.move_vertexes_array(fr_wheel, heading, xy_offset)
    fl_wheel = geometry.move_vertexes_array(fl_wheel, heading, xy_offset)
    rl_wheel = geometry.move_vertexes_array(rl_wheel, heading, xy_offset)

    vertexs_wheels = [rr_wheel, fr_wheel, fl_wheel, rl_wheel]

    Vwheels = []
    for vertexs_wheel in vertexs_wheels:
        Vwheel = geometry.ndarray_to_vertexlist(vertexs_wheel)

        Vwheels += [Vwheel]

    return Vwheels


def main():
    print("Nothing.")
    pass


if __name__ == "__main__":
    main()
