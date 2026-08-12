# ============================================================
# PI FAST SYSTEM
# Camera UDP Stream + Servo Control + Auto Scan
# ============================================================

import cv2
import socket
import threading
import time
import numpy as np
from picamera2 import Picamera2
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

# ===================== CONFIG =====================
LAPTOP_IP = "YOUR_PC_IP"
VIDEO_PORT = 9999
CONTROL_PORT = 5005

FRAME_W = 320
FRAME_H = 240

# ===================== SERVO SETUP =====================
factory = PiGPIOFactory()

servo = Servo(
    18,
    min_pulse_width=0.0005,
    max_pulse_width=0.0025,
    pin_factory=factory
)

servo_position = 0
servo.value = servo_position

oil_detected = False
previous_error = 0
integral = 0
servo_lock = threading.Lock()

# PID tuning
Kp = 0.0007
Ki = 0.0000008
Kd = 0.0004

# ===================== CAMERA STREAM THREAD =====================
def camera_stream():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (FRAME_W, FRAME_H)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    print(" Fast UDP Stream Started")

    while True:
        frame = picam2.capture_array()

        ret, buffer = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, 40]
        )

        data = buffer.tobytes()

        sock.sendto(data, (LAPTOP_IP, VIDEO_PORT))
        time.sleep(0.01)

# ===================== SERVO RECEIVER =====================
def servo_receiver():
    global previous_error, integral
    global servo_position, oil_detected

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", CONTROL_PORT))

    print("📡 Waiting for AI control data...")

    smoothed_error = 0
    alpha = 0.2

    while True:
        data, _ = sock.recvfrom(1024)
        msg = data.decode()

        if msg == "NOOIL":
            oil_detected = False
            continue

        oil_detected = True
        raw_error = int(msg)

        smoothed_error = alpha * raw_error + (1 - alpha) * smoothed_error
        error = smoothed_error

        if abs(error) < 25:
            continue

        integral += error
        integral = max(-2000, min(2000, integral))

        derivative = error - previous_error
        output = Kp*error + Ki*integral + Kd*derivative
        output = max(-0.03, min(0.03, output))

        with servo_lock:
            servo_position += output
            servo_position = max(-1, min(1, servo_position))
            servo.value = servo_position

        previous_error = error
        time.sleep(0.02)

# ===================== AUTO SCAN =====================
def auto_scan():
    global servo_position, oil_detected

    speed = 0.02
    direction = speed

    while True:
        if not oil_detected:
            with servo_lock:
                servo_position += direction

                if servo_position >= 1:
                    servo_position = 1
                    direction = -speed

                if servo_position <= -1:
                    servo_position = -1
                    direction = speed

                servo.value = servo_position

        time.sleep(0.02)

# ===================== START THREADS =====================
threading.Thread(target=camera_stream, daemon=True).start()
threading.Thread(target=servo_receiver, daemon=True).start()
threading.Thread(target=auto_scan, daemon=True).start()

while True:
    time.sleep(1)