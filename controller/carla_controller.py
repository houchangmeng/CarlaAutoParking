import carla
import numpy as np

from CarlaAutoParking.others.se2state import SE2State, SE2
from config import VehicleConfig, ParkingControlConfig, CruiseControlConfig
from typing import List, TypeVar


ShiftContext = TypeVar("ShiftContext")

SHIFT_EPS = 0.01
THROTTLE_EPS = 0.000001


class ShiftState:
    def handle(self, shiftcontext: ShiftContext):
        pass

    def get_vehicle_control(self, shiftcontext: ShiftContext):
        pass

    def __repr__(self) -> str:
        return f"ShiftState baseclass"


class StaticState(ShiftState):
    def handle(self, shiftcontext: ShiftContext):
        if shiftcontext.u[0] > THROTTLE_EPS:
            shiftcontext.set_shiftstate(ForwardAccel())
        elif shiftcontext.u[0] < -THROTTLE_EPS:
            shiftcontext.set_shiftstate(BackwardAccel())
        else:
            pass

    def get_vehicle_control(self, shiftcontext: ShiftContext):
        return carla.VehicleControl()

    def __repr__(self) -> str:
        return f"Static State"


class ForwardAccel(ShiftState):
    def handle(self, shiftcontext: ShiftContext):
        if shiftcontext.current_se2state.v < -SHIFT_EPS:
            raise ValueError("ForwardDecel value error.")

        if shiftcontext.u[0] > 0:
            pass
        else:
            shiftcontext.set_shiftstate(ForwardDecel())

    def get_vehicle_control(self, shiftcontext: ShiftContext):
        control = carla.VehicleControl()
        acceleration, delta = shiftcontext.u
        steer = delta / shiftcontext.max_delta
        control.steer = float(
            np.clip(steer, -shiftcontext.max_steer, shiftcontext.max_steer)
        )
        control.steer = float(steer)
        control.throttle = float(min(abs(acceleration), shiftcontext.max_throttle))
        control.brake = 0.0

        return control

    def __repr__(self) -> str:
        return f"ForwardAccel"


class ForwardDecel(ShiftState):
    def handle(self, shiftcontext: ShiftContext):
        if shiftcontext.current_se2state.v < -SHIFT_EPS:
            raise ValueError("ForwardDecel value error.")

        if shiftcontext.u[0] >= 0:
            shiftcontext.set_shiftstate(ForwardAccel())
        else:
            # input is negative.
            if abs(shiftcontext.current_se2state.v) > SHIFT_EPS:
                shiftcontext.set_shiftstate(ForwardDecel())
            else:
                if abs(shiftcontext.u[0]) > THROTTLE_EPS:
                    shiftcontext.set_shiftstate(BackwardAccel())
                else:
                    shiftcontext.set_shiftstate(StaticState())

    def get_vehicle_control(self, shiftcontext: ShiftContext):
        control = carla.VehicleControl()
        control.reverse = False
        acceleration, delta = shiftcontext.u
        steer = delta / shiftcontext.max_delta
        control.steer = float(
            np.clip(steer, -shiftcontext.max_steer, shiftcontext.max_steer)
        )
        control.throttle = 0.0
        control.brake = float(min(abs(acceleration), shiftcontext.max_brake))

        return control

    def __repr__(self) -> str:
        return f"ForwardDecel"


class BackwardDecel(ShiftState):
    def handle(self, shiftcontext: ShiftContext):
        if shiftcontext.current_se2state.v > SHIFT_EPS:
            raise ValueError("BackwardAccel value error.")

        if shiftcontext.u[0] < 0:
            shiftcontext.set_shiftstate(BackwardAccel())
        else:
            if abs(shiftcontext.current_se2state.v) > SHIFT_EPS:
                shiftcontext.set_shiftstate(BackwardDecel())
            else:
                if abs(shiftcontext.u[0]) > THROTTLE_EPS:
                    shiftcontext.set_shiftstate(ForwardAccel())
                else:
                    shiftcontext.set_shiftstate(StaticState())

    def get_vehicle_control(self, shiftcontext: ShiftContext):
        control = carla.VehicleControl()
        control.reverse = True
        acceleration, delta = shiftcontext.u
        steer = delta / shiftcontext.max_delta
        control.steer = float(
            np.clip(steer, -shiftcontext.max_steer, shiftcontext.max_steer)
        )
        control.throttle = 0.0
        control.brake = float(min(abs(acceleration), shiftcontext.max_brake))

        return control

    def __repr__(self) -> str:
        return f"BackwardDecel"


class BackwardAccel(ShiftState):
    def handle(self, shiftcontext: ShiftContext):
        if shiftcontext.current_se2state.v > SHIFT_EPS:
            # raise ValueError("BackwardDecel value error.")
            pass

        if shiftcontext.u[0] <= 0:
            pass
        else:
            shiftcontext.set_shiftstate(BackwardDecel())

    def get_vehicle_control(self, shiftcontext: ShiftContext):
        control = carla.VehicleControl()
        control.reverse = True
        acceleration, delta = shiftcontext.u
        steer = delta / shiftcontext.max_delta
        control.steer = float(
            np.clip(steer, -shiftcontext.max_steer, shiftcontext.max_steer)
        )
        control.throttle = float(min(abs(acceleration), shiftcontext.max_throttle))
        control.brake = 0.0

        return control

    def __repr__(self) -> str:
        return f"BackwardAccel"


class ShiftContext:
    def __init__(
        self, veh_cfg=VehicleConfig(), ctrl_cfg=ParkingControlConfig()
    ) -> None:
        self.veh_cfg = veh_cfg
        self.state: ShiftState = None
        self.current_se2state = None
        self.u = None

        self.max_delta = veh_cfg.max_front_wheel_angle
        self.max_steer = ctrl_cfg.max_steer
        self.max_brake = ctrl_cfg.max_brake
        self.max_throttle = ctrl_cfg.max_throttle

        self.set_shiftstate(StaticState())

    def action(self, current_se2state: SE2State, u) -> carla.VehicleControl:
        self.u = u
        self.current_se2state = current_se2state
        self.state.handle(self)
        return self.state.get_vehicle_control(self)

    def set_shiftstate(self, shiftstate: ShiftState):
        self.state = shiftstate


from CarlaAutoParking.controller.controller import (
    OptiTrackingController,
    LatLQRLonPIDController,
    # LatLQRLonSpeedPIDController,
    LatPIDLonSpeedPIDController,
)


class CarlaParkingController:
    def __init__(
        self,
        reference_trajectory: List[SE2State] = None,
        veh_cfg=VehicleConfig(),
        ctrl_cfg=ParkingControlConfig(),
    ) -> None:
        if reference_trajectory is not None:
            self.controller = OptiTrackingController(
                reference_trajectory,
                veh_cfg,
                ctrl_cfg,
                ctrl_cfg.controller_type,
                ctrl_cfg.horizon,
            )
        else:
            self.controller = None

        self.veh_cfg = veh_cfg
        self.ctrl_cfg = ctrl_cfg
        self.controller_type = ctrl_cfg.controller_type
        self.horizon = ctrl_cfg.horizon
        self.shift_context = ShiftContext(veh_cfg, ctrl_cfg)

    def set_reference_trajectory(self, reference_trajectory: List[SE2State]):
        self.controller = OptiTrackingController(
            reference_trajectory,
            self.veh_cfg,
            self.ctrl_cfg,
            self.controller_type,
            self.horizon,
        )

    def action(self, current_se2state: SE2State):
        if self.controller is None:
            raise ValueError("Set reference trajectory.")

        u = self.controller.action(current_se2state)
        control: carla.VehicleControl = self.shift_context.action(current_se2state, u)
        control.hand_brake = False
        control.manual_gear_shift = False
        print(self.shift_context.state)
        return control

    def emergency_stop(self):
        control = carla.VehicleControl()
        control.throttle = 0
        control.steer = 0
        control.brake = 1.0
        control.hand_brake = True

        return control


# class CarlaParkingController(OptiTrackingController):
#     def __init__(
#         self,
#         reference_trajectory: List[SE2State] = None,
#         veh_cfg=VehicleConfig(),
#         ctrl_cfg=ParkingControlConfig(),
#         controller_type="Finite_LQR",
#         horizon=20,
#     ) -> None:
#         """
#         controller_type ="Finite_LQR", "LQR", "MPC"
#         """
#         super().__init__(
#             reference_trajectory, veh_cfg, ctrl_cfg, controller_type, horizon
#         )

#         self.shift_context = ShiftContext()

#     def action(self, current_se2state: SE2State):
#         u = super().action(current_se2state)

#         """
#         State:

#         static ,
#         forward accelerate,
#         forward decelerate,
#         backward accelerate,
#         backward decelerate
#         """

#         control: carla.VehicleControl = self.shift_context.action(current_se2state, u)
#         control.hand_brake = False
#         control.manual_gear_shift = False
#         print(self.shift_context.state)
#         return control

#     def emergency_stop(self):
#         control = carla.VehicleControl()
#         control.throttle = 0
#         control.steer = 0
#         control.brake = 1.0
#         control.hand_brake = True

#         return control


# class CarlaCruiseController:
#     def __init__(
#         self,
#         reference_trajectory: List[SE2State] = None,
#         veh_cfg=VehicleConfig(),
#         ctrl_cfg=CruiseControlConfig(),
#     ) -> None:
#         if reference_trajectory is not None:
#             self.controller = LatLQRLonPIDController(
#                 reference_trajectory, veh_cfg, ctrl_cfg
#             )
#         else:
#             self.controller = None

#         self.veh_cfg = veh_cfg
#         self.ctrl_cfg = ctrl_cfg
#         self.shift_context = ShiftContext(veh_cfg, ctrl_cfg)

#     def set_reference_trajectory(self, reference_trajectory: List[SE2State]):
#         self.controller = LatLQRLonPIDController(
#             reference_trajectory, self.veh_cfg, self.ctrl_cfg
#         )

#     def action(self, current_se2state: SE2State):
#         if self.controller is None:
#             raise ValueError("Set reference trajectory.")

#         u = self.controller.action(current_se2state)
#         control: carla.VehicleControl = self.shift_context.action(current_se2state, u)
#         control.hand_brake = False
#         control.manual_gear_shift = False
#         print(self.shift_context.state)
#         return control

#     def emergency_stop(self):
#         control = carla.VehicleControl()
#         control.throttle = 0
#         control.steer = 0
#         control.brake = 1.0
#         control.hand_brake = True

#         return control


# class CarlaCruiseController(LatLQRLonPIDController):
#     def __init__(
#         self,
#         reference_trajectory: List[SE2State] = None,
#         veh_cfg=VehicleConfig(),
#         ctrl_cfg=CruiseControlConfig(),
#     ) -> None:
#         super().__init__(reference_trajectory, veh_cfg, ctrl_cfg)

#         self.shift_context = ShiftContext()

#     def action(self, current_se2state: SE2State):
#         u = super().action(current_se2state)

#         """
#         State:

#         static ,
#         forward accelerate,
#         forward decelerate,
#         backward accelerate,
#         backward decelerate
#         """

#         control: carla.VehicleControl = self.shift_context.action(current_se2state, u)
#         control.hand_brake = False
#         control.manual_gear_shift = False
#         print(self.shift_context.state)
#         return control

#     def emergency_stop(self):
#         control = carla.VehicleControl()
#         control.throttle = 0
#         control.steer = 0
#         control.brake = 1.0
#         control.hand_brake = True

#         return control


class CarlaCruiseController:
    def __init__(
        self,
        reference_trajectory: List[SE2State] = None,
        veh_cfg=VehicleConfig(),
        ctrl_cfg=CruiseControlConfig(),
    ) -> None:
        if reference_trajectory is not None:
            self.controller = LatPIDLonSpeedPIDController(
                reference_trajectory, veh_cfg, ctrl_cfg
            )
        else:
            self.controller = None

        self.veh_cfg = veh_cfg
        self.ctrl_cfg = ctrl_cfg
        self.shift_context = ShiftContext(veh_cfg, ctrl_cfg)

    def set_reference_trajectory(self, reference_trajectory: List[SE2State]):
        self.controller = LatPIDLonSpeedPIDController(
            reference_trajectory, self.veh_cfg, self.ctrl_cfg
        )

    def action(self, current_se2state: SE2State):
        if self.controller is None:
            raise ValueError("Set reference trajectory.")

        u = self.controller.action(current_se2state)
        control: carla.VehicleControl = self.shift_context.action(current_se2state, u)
        control.hand_brake = False
        control.manual_gear_shift = False
        print(self.shift_context.state)
        return control

    def emergency_stop(self):
        control = carla.VehicleControl()
        control.throttle = 0
        control.steer = 0
        control.brake = 1.0
        control.hand_brake = True

        return control
