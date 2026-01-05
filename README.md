# pi-esp32-edge-ai-rover
Raspberry Pi + ESP32 powered RC robot with mecanum wheels, edge AI face detection, and lightweight LLM voice commands. ESP32 handles real-time motor control while the Pi runs vision and speech models.

## FPV face detection stream
Run the controller with a YuNet face detection MJPEG stream on the local network:

```bash
python3 main.py --face-stream
```

The stream is served at `http://<pi-ip>:8080/` by default. Override camera and model settings with `--face-*` flags.
