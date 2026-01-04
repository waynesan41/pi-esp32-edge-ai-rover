import logging
import os

from pyPS4Controller.controller import Controller

from quit_control import QuitControl

MAX_JOY = 32767  # Raw joystick range from pyPS4Controller.
MAX_DUTY = 70  # Motor duty range for the controller board.
DEADZONE = 0.3  # Ignore small stick movements near center.
MIN_INPUT_CHANGE = 0.1  # Debounce tiny changes after normalization.

LOG_FILENAME = "wheel_speed_normalized.log"

MOTOR_PORTS = {  # Hardware port mapping for each wheel.
    "front_right": 1,
    "rear_right": 3,
    "front_left": 4,
    "rear_left": 2,
}

MOTOR_SIGNS = {  # Flip sign to correct wheel direction.
    "front_left": -1,
    "front_right": 1,
    "rear_left": -1,
    "rear_right": 1,
}


def clamp(value, lo, hi):
    # Keep values inside a predictable range.
    return max(lo, min(hi, value))


class MotorControl:
    """Compute mecanum wheel duties and send them to the motor controller."""

    def __init__(self, board):
        self._board = board
        # Raw left stick state.
        self._lx = 0
        self._ly = 0
        # Raw trigger state for rotation.
        self._l2 = 0
        self._r2 = 0
        # Cached normalized values to reduce jitter.
        self._last_norm = (0.0, 0.0, 0.0)
        self._last_duties = None
        self._logger = self._init_logger()

    def _init_logger(self):
        # Create a single file-backed logger for wheel speed telemetry.
        log_path = os.path.join(os.path.dirname(__file__), LOG_FILENAME)
        logger = logging.getLogger("wheel_speed")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter(
                "%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _normalize(self, value):
        # Map raw stick values to [-1, 1] with a deadzone.
        if abs(value) < DEADZONE * MAX_JOY:
            return 0.0
        return clamp(value / MAX_JOY, -1.0, 1.0)

    def set_stick(self, x=None, y=None):
        # Update cached stick values then recompute motor duties.
        if x is not None:
            self._lx = x
        if y is not None:
            self._ly = y
        self._update()

    def set_trigger(self, l2=None, r2=None):
        # Update trigger values then recompute motor duties.
        if l2 is not None:
            self._l2 = l2
        if r2 is not None:
            self._r2 = r2
        self._update()

    def _normalize_trigger(self, value):
        # Ignore negative trigger input; map positive values to [0, 1].
        if value <= 0:
            return 0.0
        return clamp(value / MAX_JOY, 0.0, 1.0)

    def _update(self):
        # Convert raw inputs into normalized strafe/forward/rotation intent.
        # Left stick only: strafe (x) and forward (y).
        x = self._normalize(self._lx)
        y = -self._normalize(self._ly)  # stick up -> forward
        l2 = self._normalize_trigger(self._l2)
        r2 = self._normalize_trigger(self._r2)
        r = r2 - l2

        # Ignore small post-deadzone changes to reduce jitter.
        if (
            abs(x - self._last_norm[0]) < MIN_INPUT_CHANGE
            and abs(y - self._last_norm[1]) < MIN_INPUT_CHANGE
            and abs(r - self._last_norm[2]) < MIN_INPUT_CHANGE
        ):
            return
        self._last_norm = (x, y, r)

        if l2 > 0.0 and r2 > 0.0:
            # Hard stop when both triggers are pressed.
            front_left = 0.0
            front_right = 0.0
            rear_left = 0.0
            rear_right = 0.0
        else:
            # Standard mecanum mixing for strafe (x), forward (y), and rotate (r).
            front_left = y + x + r
            front_right = y - x - r
            rear_left = y - x + r
            rear_right = y + x - r

        # Normalize so the max magnitude is 1.0 to preserve direction.
        max_mag = max(
            1.0,
            abs(front_left),
            abs(front_right),
            abs(rear_left),
            abs(rear_right),
        )
        front_left /= max_mag
        front_right /= max_mag
        rear_left /= max_mag
        rear_right /= max_mag

        # Convert normalized wheel values into motor duty commands.
        duties = [
            [
                MOTOR_PORTS["front_left"],
                int(front_left * MAX_DUTY * MOTOR_SIGNS["front_left"]),
            ],
            [
                MOTOR_PORTS["front_right"],
                int(front_right * MAX_DUTY * MOTOR_SIGNS["front_right"]),
            ],
            [
                MOTOR_PORTS["rear_left"],
                int(rear_left * MAX_DUTY * MOTOR_SIGNS["rear_left"]),
            ],
            [
                MOTOR_PORTS["rear_right"],
                int(rear_right * MAX_DUTY * MOTOR_SIGNS["rear_right"]),
            ],
        ]

        if duties != self._last_duties:
            self._logger.info(
                "norm FL=%.3f FR=%.3f RL=%.3f RR=%.3f duty FL=%d FR=%d RL=%d RR=%d",
                front_left,
                front_right,
                rear_left,
                rear_right,
                duties[0][1],
                duties[1][1],
                duties[2][1],
                duties[3][1],
            )
            # Only send updates when something changed.
            self._board.set_motor_duty(duties)
            self._last_duties = duties

    def stop(self):
        # Hard stop all motors.
        self._board.set_motor_duty(
            [
                [MOTOR_PORTS["front_left"], 0],
                [MOTOR_PORTS["front_right"], 0],
                [MOTOR_PORTS["rear_left"], 0],
                [MOTOR_PORTS["rear_right"], 0],
            ]
        )
        self._last_duties = None


class MecanumPS4Controller(Controller, QuitControl):
    def __init__(self, board, **kwargs):
        # Wire the PS4 controller to motor control, passing through library args.
        super().__init__(**kwargs)
        # Motor handler for left-stick mecanum driving.
        self._motor = MotorControl(board)
        # Enable Options+Share quit combo handling.
        QuitControl.__init__(self)

    def stop_motors(self):
        # Stop motors on command or when exiting.
        self._motor.stop()

    def on_L3_up(self, value):
        # Forward stick motion.
        self._motor.set_stick(y=value)

    def on_L3_down(self, value):
        # Backward stick motion.
        self._motor.set_stick(y=value)

    def on_L3_left(self, value):
        # Left strafe.
        self._motor.set_stick(x=value)

    def on_L3_right(self, value):
        # Right strafe.
        self._motor.set_stick(x=value)

    def on_L3_at_rest(self):
        # Stop motion when the stick centers.
        self._motor.set_stick(x=0, y=0)

    def on_L3_x_at_rest(self):
        # Clear strafe only.
        self._motor.set_stick(x=0)

    def on_L3_y_at_rest(self):
        # Clear forward/back only.
        self._motor.set_stick(y=0)

    def on_L2_press(self, value):
        # Rotate left with L2 (ignore negative values).
        self._motor.set_trigger(l2=value)

    def on_L2_release(self):
        self._motor.set_trigger(l2=0)

    def on_L2_at_rest(self):
        self._motor.set_trigger(l2=0)

    def on_R2_press(self, value):
        # Rotate right with R2 (ignore negative values).
        self._motor.set_trigger(r2=value)

    def on_R2_release(self):
        self._motor.set_trigger(r2=0)

    def on_R2_at_rest(self):
        self._motor.set_trigger(r2=0)

    def on_options_press(self):
        # Safety stop and check quit combo.
        QuitControl.on_options_press(self)
        self.stop_motors()

    def on_options_release(self):
        QuitControl.on_options_release(self)

    def on_share_press(self):
        # Exit app when Options and Share are pressed together.
        QuitControl.on_share_press(self)

    def on_share_release(self):
        QuitControl.on_share_release(self)
