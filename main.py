import argparse

# Vendor SDK for the motor controller board.
import HiwonderSDK.ros_robot_controller_sdk as rrc

# Local modules for face streaming and mecanum control.
from face_stream import DEFAULT_MODEL, FaceStreamRunner
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
    parser.add_argument(
        "--face-stream",
        action="store_true",
        help="Enable YuNet face detection MJPEG streaming.",
    )
    # Face stream configuration (camera + detector + MJPEG server).
    parser.add_argument("--face-model", default=DEFAULT_MODEL)
    parser.add_argument("--face-device", default="/dev/video0")
    parser.add_argument("--face-width", type=int, default=0)
    parser.add_argument("--face-height", type=int, default=0)
    parser.add_argument("--face-score-threshold", type=float, default=0.9)
    parser.add_argument("--face-nms-threshold", type=float, default=0.3)
    parser.add_argument("--face-top-k", type=int, default=5000)
    parser.add_argument("--face-draw-landmarks", action="store_true")
    parser.add_argument("--face-stream-host", default="0.0.0.0")
    parser.add_argument("--face-stream-port", type=int, default=8080)
    parser.add_argument("--face-stream-fps", type=float, default=10.0)
    args = parser.parse_args()

    # Initialize the motor controller board and PS4 controller listener.
    board = rrc.Board()
    controller = MecanumPS4Controller(
        board,
        interface=args.interface,
        connecting_using_ds4drv=args.ds4drv,
    )
    face_runner = None
    if args.face_stream:
        # Start the YuNet detector and MJPEG server in the background.
        face_runner = FaceStreamRunner(
            model_path=args.face_model,
            device=args.face_device,
            width=args.face_width,
            height=args.face_height,
            score_threshold=args.face_score_threshold,
            nms_threshold=args.face_nms_threshold,
            top_k=args.face_top_k,
            draw_landmarks=args.face_draw_landmarks,
            stream_host=args.face_stream_host,
            stream_port=args.face_stream_port,
            stream_fps=args.face_stream_fps,
        )
        try:
            face_runner.start()
        except (RuntimeError, FileNotFoundError) as exc:
            # If OpenCV or the model is missing, continue without streaming.
            print(f"Face stream disabled: {exc}")
            face_runner = None
    try:
        # Listen for controller events until timeout or interrupt.
        controller.listen(timeout=60)
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C without a traceback.
        pass
    finally:
        # Always stop the motors on exit.
        controller.stop_motors()
        if face_runner is not None:
            # Ensure the stream thread shuts down as well.
            face_runner.stop()


if __name__ == "__main__":
    main()
