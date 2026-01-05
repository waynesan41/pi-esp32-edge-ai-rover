import argparse
import http.server
import os
import socketserver
import threading
import time

try:
    import cv2
except ImportError:
    cv2 = None


# Default YuNet model path relative to the project root.
DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "face_detection_yunet_2023mar.onnx",
)


# Threaded HTTP server to serve a continuous MJPEG stream.
class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _start_mjpeg_server(host, port, get_frame, fps):
    # MJPEG uses a boundary string to delimit individual JPEG frames.
    boundary = "frame"

    class MJPEGHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            # Only serve the root or /stream endpoints.
            if self.path not in ("/", "/stream"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={boundary}",
            )
            self.end_headers()

            # Use a fixed delay to pace the stream if fps is set.
            delay = 1.0 / fps if fps > 0 else 0.0
            while True:
                # Pull the latest frame snapshot from the producer thread.
                frame = get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                # Encode the frame as JPEG for MJPEG streaming.
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                try:
                    # Write the multipart MJPEG frame with headers.
                    self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(encoded.tobytes())
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                if delay:
                    time.sleep(delay)

        def log_message(self, format, *args):
            # Suppress default HTTP request logging.
            return

    server = _ThreadingHTTPServer((host, port), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _draw_faces(frame, faces, draw_landmarks):
    # Draw bounding boxes, confidence score, and optional landmarks.
    for face in faces:
        x, y, w, h = face[:4]
        score = face[4]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{score:.2f}",
            (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        if draw_landmarks and len(face) >= 15:
            for i in range(5):
                lx = int(face[5 + i * 2])
                ly = int(face[5 + i * 2 + 1])
                cv2.circle(frame, (lx, ly), 2, (0, 255, 255), -1)


class FaceStreamRunner:
    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        device="/dev/video0",
        width=0,
        height=0,
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000,
        draw_landmarks=False,
        stream_host="0.0.0.0",
        stream_port=8080,
        stream_fps=10.0,
        display=False,
    ):
        # Capture configuration and model parameters.
        self._model_path = model_path
        self._device = device
        self._width = width
        self._height = height
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._draw_landmarks = draw_landmarks
        self._stream_host = stream_host
        self._stream_port = stream_port
        self._stream_fps = stream_fps
        self._display = display

        # Runtime state (threading, frame cache, and HTTP server).
        self._thread = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._server = None

    def start(self):
        # Start the capture + detection thread if it is not running yet.
        if self._thread is not None:
            return
        if cv2 is None:
            raise RuntimeError("OpenCV not installed.")
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"YuNet model not found: {self._model_path}")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        # Signal the worker thread to stop and wait for a clean shutdown.
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _get_latest_frame(self):
        # Return a copy so the HTTP server never mutates the shared frame.
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _run(self):
        # Convert numeric device strings (e.g., "0") into camera indexes.
        device = int(self._device) if str(self._device).isdigit() else self._device
        cap = cv2.VideoCapture(device)
        if self._width > 0 and self._height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # Grab one frame to validate the capture and configure the detector.
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return

        height, width = frame.shape[:2]
        # Create the YuNet detector with the initial input size.
        detector = cv2.FaceDetectorYN.create(
            self._model_path,
            "",
            (width, height),
            self._score_threshold,
            self._nms_threshold,
            self._top_k,
        )

        window_name = "YuNet Face Detection"
        # Start the HTTP MJPEG server in a background thread.
        self._server = _start_mjpeg_server(
            self._stream_host,
            self._stream_port,
            self._get_latest_frame,
            self._stream_fps,
        )
        print(f"Streaming on http://{self._stream_host}:{self._stream_port}/")

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                # Update the detector input size for any resolution changes.
                height, width = frame.shape[:2]
                detector.setInputSize((width, height))
                _, faces = detector.detect(frame)
                if faces is not None and len(faces):
                    _draw_faces(frame, faces, self._draw_landmarks)
                # Update the shared frame for the HTTP server.
                with self._frame_lock:
                    self._latest_frame = frame
                if self._display:
                    # Optional local preview window with exit on Esc or "q".
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
        finally:
            # Always release the camera and shut down the HTTP server.
            cap.release()
            if self._server is not None:
                self._server.shutdown()
                self._server = None
            if self._display:
                cv2.destroyAllWindows()


def main():
    # Parse command-line options for the camera, detector, and stream settings.
    parser = argparse.ArgumentParser(
        description="Stream YuNet face detection over MJPEG."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--draw-landmarks", action="store_true")
    parser.add_argument("--stream-host", default="0.0.0.0")
    parser.add_argument("--stream-port", type=int, default=8080)
    parser.add_argument("--stream-fps", type=float, default=10.0)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    # Disable display output when running without a GUI session.
    if args.display and not os.environ.get("DISPLAY"):
        print("No DISPLAY detected; disabling window display.")
        args.display = False

    # Spin up the face stream runner in a background thread.
    runner = FaceStreamRunner(
        model_path=args.model,
        device=args.device,
        width=args.width,
        height=args.height,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        draw_landmarks=args.draw_landmarks,
        stream_host=args.stream_host,
        stream_port=args.stream_port,
        stream_fps=args.stream_fps,
        display=args.display,
    )
    try:
        runner.start()
        # Keep the process alive until interrupted.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure the worker thread stops cleanly.
        runner.stop()


if __name__ == "__main__":
    main()
