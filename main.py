import argparse

import HiwonderSDK.ros_robot_controller_sdk as rrc

from wheel_control import MecanumPS4Controller


def main():
    # Entry point for the PS4 mecanum control app.
    parser = argparse.ArgumentParser(
        description="PS4 left-stick mecanum controller (isolated app)."
    )
    # Controller device path (change if your joystick device is different).
    parser.add_argument("--interface", default="/dev/input/js0")
    parser.add_argument(
        "--ds4drv",
        action="store_true",
        help="Set if the controller is connected via ds4drv.",
    )
    args = parser.parse_args()

    # Initialize the motor controller board and PS4 controller listener.
    board = rrc.Board()
    controller = MecanumPS4Controller(
        board,
        interface=args.interface,
        connecting_using_ds4drv=args.ds4drv,
    )
    try:
        # Listen for controller events until timeout or interrupt.
        controller.listen(timeout=60)
    finally:
        # Always stop the motors on exit.
        controller.stop_motors()


if __name__ == "__main__":
    main()
