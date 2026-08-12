import cv2, threading, time, queue, sys, os, socket
import requests, numpy as np, json
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# =============================================================
#  CONFIGURATION — edit these three values before running
# =============================================================
PI_IP       = "ENTER_PI_IP"   # Raspberry Pi IP address
PC_IP       = "ENTER_PC_IP"   # This PC's IP address (Pi streams camera here)
ENGINE_PATH = r"ENTER_MODEL_PATH(TensorRt)"    # Path to TensorRT engine file
# =============================================================

PI_PORT        = 8080
UDP_CAM_PORT   = 9999
UDP_SERVO_PORT = 5005
WEB_PORT       = 5000

OIL_WARMUP_RUNS = 10
OIL_IMG_SIZE    = 640

OIL_CLASS_COLORS_BGR = {
    0: (30,  30,  30),
    1: (40, 110, 170),
}
OIL_DEFAULT_COLOR = (0, 200, 80)

FRAME_W, FRAME_H  = 640, 480
OIL_CONF          = 0.65
OIL_IOU           = 0.45
OIL_MIN_AREA_FRAC = 0.003
OIL_MASK_ALPHA    = 0.40
OIL_DEAD_ZONE     = 15
OIL_SEND_RATE     = 0.02
PROXIMITY_RATIO   = 0.12
CMD_COOLDOWN      = 0.3
LEFT_ZONE_END     = FRAME_W // 3
RIGHT_ZONE_START  = 2 * FRAME_W // 3

OIL_CLASSES = None


class SharedState:
    def __init__(self):
        self._l = threading.Lock()
        self.mode             = "MANUAL"
        self.roller_on        = False
        self.roller_pwm       = 80
        self.pump_on          = False
        self.pump_pwm         = 80
        self.drive_speed      = 200
        self.last_cmd         = "---"
        self.pi_ok            = False
        self.pi_status        = "Waiting..."
        self.cam_ok           = False
        self.fps              = 0.0
        self.joy_x            = 0.0
        self.joy_y            = 0.0
        self.joy_active       = False
        self.oil_tracking     = False
        self.oil_error        = 0
        self.oil_coverage_pct = 0.0
        self.pan_pos          = 90
        self.tilt_pos         = 90
        self.oil_infer_ms     = 0.0

    def get(self, k):
        with self._l: return getattr(self, k)

    def set(self, k, v):
        with self._l: setattr(self, k, v)


S = SharedState()


class PiHttp(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.base     = f"http://{PI_IP}:{PI_PORT}"
        self._q       = queue.Queue(maxsize=10)
        self._stop    = threading.Event()
        self._session = requests.Session()

    def cmd(self, path):
        try: self._q.put_nowait(("GET", path))
        except queue.Full: pass

    def pwm(self, l, r):      self._fire(f"/pwm?l={int(l)}&r={int(r)}")
    def roller_pwm(self, dc): self._fire(f"/roller_pwm?v={int(dc)}")
    def pump_pwm(self, dc):   self._fire(f"/pump_pwm?v={int(dc)}")
    def speed(self, v):       self._fire(f"/speed?v={int(v)}")
    def pan_set(self, deg):   self._fire(f"/pan_set?v={int(deg)}")
    def tilt_set(self, deg):  self._fire(f"/tilt_set?v={int(deg)}")

    def _fire(self, path):
        try: self._q.put_nowait(("GET", path))
        except queue.Full: pass

    def _get(self, path):
        try:
            r = self._session.get(self.base + path, timeout=1.2)
            S.set("pi_ok",     r.status_code == 200)
            S.set("pi_status", path.split("?")[0] if r.status_code == 200
                               else f"HTTP {r.status_code}")
        except requests.exceptions.ConnectionError:
            S.set("pi_ok", False); S.set("pi_status", "Connection refused")
        except requests.exceptions.Timeout:
            S.set("pi_ok", False); S.set("pi_status", "Timeout")
        except Exception as e:
            S.set("pi_ok", False); S.set("pi_status", str(e)[:30])

    def stop_thread(self): self._stop.set(); self.cmd("/stop")

    def run(self):
        self._get("/stop")
        while not self._stop.is_set():
            try:
                _, path = self._q.get(timeout=0.1); self._get(path)
            except queue.Empty: pass
        self._get("/stop")


pi = PiHttp()
cam_ref = [None]


class UDPCamera(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame      = None
        self.frame_lock = threading.Lock()
        self._stop      = threading.Event()
        self.sock       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind(("0.0.0.0", UDP_CAM_PORT))
        self.sock.settimeout(3.0)
        print(f"[CAMERA] UDP :{UDP_CAM_PORT}")

    def read(self):
        with self.frame_lock:
            f = self.frame
            return (True, f.copy()) if f is not None else (False, None)

    def stop(self): self._stop.set()

    def run(self):
        current_seq    = None
        current_total  = None
        current_chunks = {}
        fc = 0; ft = time.time()

        while not self._stop.is_set():
            try:
                raw, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                S.set("cam_ok", False)
                current_seq = None; current_total = None; current_chunks = {}
                continue
            except Exception as e:
                print(f"[CAMERA] recv error: {e}"); time.sleep(0.05); continue

            try:
                p1    = raw.index(b"|")
                p2    = raw.index(b"|", p1 + 1)
                p3    = raw.index(b"|", p2 + 1)
                seq   = int(raw[:p1])
                total = int(raw[p1 + 1:p2])
                idx   = int(raw[p2 + 1:p3])
                chunk = raw[p3 + 1:]
            except (ValueError, IndexError):
                continue

            if seq != current_seq:
                current_seq    = seq
                current_total  = total
                current_chunks = {}

            current_chunks[idx] = chunk

            if len(current_chunks) == current_total:
                jpeg_bytes    = b"".join(current_chunks[i] for i in range(current_total))
                current_seq   = None; current_total = None; current_chunks = {}
                arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                    with self.frame_lock: self.frame = frame
                    S.set("cam_ok", True)
                    fc += 1
                    if time.time() - ft >= 10.0:
                        print(f"[CAMERA] {fc / (time.time() - ft):.1f} fps")
                        fc = 0; ft = time.time()


def joy_to_pwm(x, y, max_dc=100):
    x = max(-1.0, min(1.0, float(x))); y = max(-1.0, min(1.0, float(y)))
    l = y + x; r = y - x
    mag = max(abs(l), abs(r), 1.0); l /= mag; r /= mag
    return round(l * max_dc), round(r * max_dc)


def joystick_loop():
    px, py = 0.0, 0.0
    while True:
        if S.get("mode") == "MANUAL" and S.get("joy_active"):
            x, y = S.get("joy_x"), S.get("joy_y")
            if x != px or y != py:
                px, py = x, y
                l, r = joy_to_pwm(x, y, S.get("drive_speed") / 255 * 100)
                pi.pwm(l, r); S.set("last_cmd", f"JOY L{l:+d} R{r:+d}")
        elif not S.get("joy_active") and (px != 0 or py != 0):
            px, py = 0.0, 0.0; pi.cmd("/stop")
        time.sleep(0.04)


def _is_oil(model, class_id):
    global OIL_CLASSES
    if OIL_CLASSES is None:
        OIL_CLASSES = [i for i, n in model.names.items()
                       if any(kw in n.lower() for kw in ('oil', 'spill', 'dense'))]
        if OIL_CLASSES:
            print(f"[OIL] Tracking classes: {[model.names[i] for i in OIL_CLASSES]}")
        else:
            print("[OIL] WARNING: no oil/spill/dense class found — accepting all")
            OIL_CLASSES = list(model.names.keys())
    return class_id in OIL_CLASSES


class Smoother:
    def __init__(self, cd): self.cd = cd; self.last = None; self.t = 0.0

    def should(self, cmd):
        now = time.time()
        if cmd != self.last or now - self.t >= self.cd:
            self.last = cmd; self.t = now; return True
        return False


def decide(dets):
    if not dets: return "STOP"
    best = max(dets, key=lambda d: d["conf"])
    if best["area"] / (FRAME_W * FRAME_H) >= PROXIMITY_RATIO: return "COLLECT"
    if best["cx"] < LEFT_ZONE_END:    return "LEFT"
    if best["cx"] > RIGHT_ZONE_START: return "RIGHT"
    return "FORWARD"


# ── Web UI ────────────────────────────────────────────────────
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no,viewport-fit=cover">
<title>Autonomous Boat</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root[data-theme="dark"]{--bg:#0a0a0a;--surface:#111;--border:#222;--text:#fff;--muted:#999;--dim:#2a2a2a;--accent:#cc2222;--acc-dim:rgba(204,34,34,0.18);--green:#22cc55}
:root[data-theme="light"]{--bg:#f0f0f0;--surface:#fff;--border:#ddd;--text:#000;--muted:#666;--dim:#ccc;--accent:#cc2222;--acc-dim:rgba(204,34,34,0.12);--green:#1a9940}
[data-theme="dark"] *,[data-theme="dark"] *::before,[data-theme="dark"] *::after{font-weight:700!important;color:#fff!important}
[data-theme="light"] *,[data-theme="light"] *::before,[data-theme="light"] *::after{font-weight:700!important;color:#000!important}
[data-theme="dark"] canvas,[data-theme="light"] canvas,
[data-theme="dark"] img,[data-theme="light"] img{color:transparent!important}
[data-theme="dark"] input[type=range],[data-theme="light"] input[type=range]{color:transparent!important}
[data-theme="dark"] input[type=range]::-webkit-slider-thumb,
[data-theme="light"] input[type=range]::-webkit-slider-thumb{background:#cc2222!important;border:none!important;box-shadow:0 0 0 3px rgba(204,34,34,.25)!important}
[data-theme="dark"] input[type=range]::-moz-range-thumb,
[data-theme="light"] input[type=range]::-moz-range-thumb{background:#cc2222!important;border:none!important}
[data-theme="dark"] .dot,[data-theme="light"] .dot{color:transparent!important}
[data-theme="dark"] .dot.ok,[data-theme="light"] .dot.ok{background:var(--green)!important}
[data-theme="dark"] .dot.err,[data-theme="light"] .dot.err{background:var(--accent)!important}
[data-theme="dark"] .toggle-btn:not(.active),[data-theme="light"] .toggle-btn:not(.active){color:var(--muted)!important}
[data-theme="dark"] .toggle-btn.active,[data-theme="light"] .toggle-btn.active{color:#fff!important;background:var(--accent)!important}
[data-theme="dark"] .roller-btn.active,[data-theme="light"] .roller-btn.active,
[data-theme="dark"] .conv-btn.active,[data-theme="light"] .conv-btn.active{color:var(--accent)!important}
[data-theme="dark"] .dpad-btn.held,[data-theme="light"] .dpad-btn.held{color:#fff!important}
.cmd-display{color:var(--accent)!important}
.section-label,.cam-label,.cam-status,.joy-label,.joy-vals,.vslider-label,footer{color:var(--muted)!important}
.oil-status{color:var(--accent)!important}
html,body{height:100%;background:var(--bg);font-family:"Consolas","Courier New",monospace;font-size:13px;overflow:hidden;touch-action:none;-webkit-font-smoothing:antialiased}
.layout{display:grid;grid-template-rows:42px 1fr 28px;height:100dvh}
header{display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--border);background:var(--surface);gap:10px}
.title{font-size:10px;letter-spacing:3px;text-transform:uppercase;white-space:nowrap}
.header-right{display:flex;align-items:center;gap:12px}
.status-row{display:flex;gap:10px;align-items:center;font-size:10px}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--dim);margin-right:3px;vertical-align:middle}
.theme-btn{background:none;border:1px solid var(--border);font:inherit;font-size:10px;padding:3px 8px;cursor:pointer;border-radius:2px;letter-spacing:1px}
.theme-btn:hover{border-color:var(--accent)}
footer{border-top:1px solid var(--border);padding:0 14px;background:var(--surface);display:flex;justify-content:space-between;align-items:center;font-size:10px}
.main{display:grid;grid-template-columns:220px 1fr;overflow:hidden}
.left-panel{display:flex;flex-direction:column;border-right:1px solid var(--border);overflow-y:auto;padding:12px;gap:10px}
.section{display:flex;flex-direction:column;gap:6px}
.section-label{font-size:9px;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:4px}
.toggle-row{display:flex;border:1px solid var(--border);border-radius:2px;overflow:hidden}
.toggle-btn{flex:1;padding:8px 4px;font:inherit;font-size:10px;letter-spacing:1px;background:none;border:none;cursor:pointer;transition:background .15s}
.cmd-display{font-size:18px;letter-spacing:1px}
.slider-row{display:flex;align-items:center;gap:8px}
.slider-val{font-size:12px;min-width:36px;text-align:right}
input[type=range].horizontal{-webkit-appearance:none;appearance:none;flex:1;height:3px;border-radius:2px;background:var(--dim);outline:none;cursor:pointer}
input[type=range].horizontal::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#cc2222;cursor:pointer}
.roller-btn{width:100%;padding:9px;font:inherit;font-size:10px;letter-spacing:2px;background:var(--surface);border:1px solid var(--border);cursor:pointer;border-radius:2px;transition:all .15s}
.roller-btn.active{border-color:var(--accent);background:var(--acc-dim)}
.right-panel{display:flex;flex-direction:column;overflow:hidden}
.cam-section{display:flex;flex-direction:column;align-items:center;padding:8px;gap:4px;border-bottom:1px solid var(--border);flex:3}
.cam-label{font-size:9px;letter-spacing:2px;text-transform:uppercase;align-self:flex-start}
.cam-status{font-size:10px;align-self:flex-start}
#camPreview{width:100%;flex:1;min-height:0;object-fit:contain;background:#000;border:1px solid var(--border);border-radius:2px;display:block}
.joy-section{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:6px;gap:5px}
.joy-label{font-size:9px;letter-spacing:2px;text-transform:uppercase}
#joyCanvas{display:block;touch-action:none;cursor:crosshair}
.joy-vals{font-size:10px;letter-spacing:1px}
.vslider-panel{display:none;flex-direction:column;align-items:center;justify-content:center;gap:4px;padding:6px 4px}
.vslider-label{font-size:8px;letter-spacing:1px;text-transform:uppercase;text-align:center}
.vslider-val{font-size:13px;text-align:center}
.vslider-wrap{display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;width:100%}
input[type=range].vertical{-webkit-appearance:slider-vertical;appearance:slider-vertical;writing-mode:vertical-lr;direction:rtl;width:4px;flex:1;background:var(--dim);outline:none;cursor:pointer;border-radius:2px}
input[type=range].vertical::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#cc2222;cursor:pointer}
.dpad-panel{display:none;flex-direction:column;gap:4px}
.dpad-grid{display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,1fr);gap:4px;flex:1}
.dpad-btn{background:var(--surface);border:1px solid var(--border);border-radius:3px;display:flex;align-items:center;justify-content:center;font:inherit;font-size:10px;letter-spacing:1px;cursor:pointer;user-select:none;touch-action:none;transition:background .1s,border-color .1s}
.dpad-btn.held,.dpad-btn:active{background:var(--acc-dim);border-color:var(--accent)}
.dpad-empty{background:none;border:none;cursor:default}
.conv-btn{padding:6px 4px;font:inherit;font-size:9px;letter-spacing:1px;background:var(--surface);border:1px solid var(--border);cursor:pointer;border-radius:2px;text-align:center;transition:all .15s}
.conv-btn.active{border-color:var(--accent);background:var(--acc-dim)}
.mode-mini{display:flex;border:1px solid var(--border);border-radius:2px;overflow:hidden}
.mode-mini .toggle-btn{padding:5px 2px;font-size:9px}
.joy-land{display:none;flex-direction:column;align-items:center;justify-content:center;padding:6px;gap:4px}
.joy-land .joy-label{font-size:8px;letter-spacing:2px;text-transform:uppercase}
.joy-land .joy-vals{font-size:9px}
#joyCanvasL{display:block;touch-action:none;cursor:crosshair}
.pan-row{display:flex;align-items:center;gap:6px;margin-top:4px}
.pan-val{font-size:11px;min-width:32px;text-align:right}
@media (orientation:landscape) and (max-height:540px){
  header{display:none}footer{display:none}
  .layout{grid-template-rows:1fr;height:100dvh}
  .main{display:grid;grid-template-columns:50px 165px 1fr 160px 50px;grid-template-rows:1fr;overflow:hidden;height:100dvh}
  .joy-section{display:none}
  .left-panel{display:flex;grid-column:2;flex-direction:column;border-left:1px solid var(--border);padding:6px 8px;gap:5px;overflow-y:auto}
  .left-panel .hide-landscape{display:none}
  .dpad-panel{display:flex;flex:1;min-height:0}
  .right-panel{grid-column:3;flex-direction:column;border-left:1px solid var(--border);border-right:1px solid var(--border)}
  .cam-section{flex:1;border-bottom:none;padding:4px;justify-content:center;align-items:center}
  #camPreview{width:100%;height:100%;max-height:100%;min-height:unset;flex:unset;object-fit:contain}
  .cam-label{font-size:8px}.cam-status{font-size:8px}
  .joy-land{display:flex;grid-column:4;border-right:1px solid var(--border)}
  .vslider-panel{display:flex}
  .vslider-drive{grid-column:1;border-right:1px solid var(--border)}
  .vslider-conv{grid-column:5;border-left:1px solid var(--border)}
  .section-label{font-size:7px}.toggle-btn{font-size:9px;padding:5px 2px}
  .cmd-display{font-size:12px}.dpad-btn{font-size:9px}
  .vslider-label{font-size:7px}.vslider-val{font-size:11px}
  .mode-mini .toggle-btn{font-size:8px;padding:4px 2px}
}
</style>
</head>
<body>
<div class="layout">
<header>
  <span class="title">Autonomous Boat</span>
  <div class="header-right">
    <div class="status-row">
      <span><span class="dot" id="piDot"></span>Pi</span>
      <span><span class="dot" id="camDot"></span>Cam</span>
      <span id="fpsDisplay">0.0 fps</span>
      <span id="modeDisplay">MANUAL</span>
      <span class="oil-status" id="oilStatus">OIL:SCAN</span>
    </div>
    <button class="theme-btn" id="themeBtn" onclick="toggleTheme()">LIGHT</button>
  </div>
</header>
<div class="main">
  <!-- Drive vslider col 1 -->
  <div class="vslider-panel vslider-drive">
    <div class="vslider-wrap">
      <div class="vslider-label">Drive</div>
      <div class="vslider-val" id="spdValL">200</div>
      <input type="range" class="vertical" id="spdSliderL" min="0" max="255" value="200" oninput="onSpeed(this.value)">
    </div>
  </div>
  <!-- Left panel -->
  <div class="left-panel">
    <div class="section hide-landscape">
      <div class="section-label">Command</div>
      <div class="cmd-display" id="cmdDisplay">---</div>
    </div>
    <div class="section hide-landscape">
      <div class="section-label">Mode</div>
      <div class="toggle-row">
        <button class="toggle-btn active" id="tabManual" onclick="setMode('MANUAL')">MANUAL</button>
        <button class="toggle-btn" id="tabAuto" onclick="setMode('AUTO')">AUTONOMOUS</button>
      </div>
    </div>
    <div class="section hide-landscape">
      <div class="section-label">Drive Speed</div>
      <div class="slider-row">
        <span class="slider-val" id="spdVal">200</span>
        <input type="range" class="horizontal" id="spdSlider" min="0" max="255" value="200" oninput="onSpeed(this.value)">
      </div>
    </div>
    <div class="section hide-landscape">
      <div class="section-label">Conveyor PWM</div>
      <div class="slider-row">
        <span class="slider-val" id="rolVal">80</span>
        <input type="range" class="horizontal" id="rolSlider" min="0" max="100" value="80" oninput="onRollerPwm(this.value)">
      </div>
      <button class="roller-btn" id="rollerBtn" onclick="toggleRoller()">CONVEYOR ON</button>
    </div>
    <div class="section hide-landscape">
      <div class="section-label">Pump PWM</div>
      <div class="slider-row">
        <span class="slider-val" id="pmpVal">80</span>
        <input type="range" class="horizontal" id="pmpSlider" min="0" max="100" value="80" oninput="onPumpPwm(this.value)">
      </div>
      <button class="roller-btn" id="pumpBtn" onclick="togglePump()">PUMP ON</button>
    </div>
    <div class="section hide-landscape">
      <div class="section-label">Pan Servo <span id="panModeTag" style="font-size:8px;letter-spacing:1px;color:var(--green)">MANUAL</span></div>
      <div class="slider-row">
        <span class="slider-val" id="panVal">90</span>
        <input type="range" class="horizontal" id="panSlider" min="0" max="180" value="90" oninput="onPan(this.value)">
      </div>
      <button class="roller-btn" id="panCentreBtn" onclick="setPanCentre()">CENTRE</button>
      <div style="font-size:9px;color:var(--muted);margin-top:3px" id="panNote">MANUAL: drag to aim servo</div>
    </div>
    <div class="section hide-landscape">
      <div class="section-label">Tilt Servo</div>
      <div class="slider-row">
        <span class="slider-val" id="tiltVal">90</span>
        <input type="range" class="horizontal" id="tiltSlider" min="0" max="180" value="90" oninput="onTilt(this.value)">
      </div>
      <button class="roller-btn" id="tiltCentreBtn" onclick="setTiltCentre()">LEVEL</button>
      <div style="font-size:9px;color:var(--muted);margin-top:3px" id="tiltNote">MANUAL: drag to tilt camera</div>
    </div>
    <div class="section hide-landscape">
      <div class="section-label">Pi Status</div>
      <div style="font-size:10px" id="piStatus">---</div>
    </div>
    <!-- D-pad landscape only -->
    <div class="dpad-panel">
      <div class="mode-mini">
        <button class="toggle-btn active" id="tabManualD" onclick="setMode('MANUAL')">MANUAL</button>
        <button class="toggle-btn" id="tabAutoD" onclick="setMode('AUTO')">AUTO</button>
      </div>
      <div class="dpad-grid">
        <div class="dpad-empty"></div>
        <button class="dpad-btn" ontouchstart="dpadOn('FORWARD',this)" ontouchend="dpadOff(this)" onmousedown="dpadOn('FORWARD',this)" onmouseup="dpadOff(this)" onmouseleave="dpadOff(this)">FWD</button>
        <div class="dpad-empty"></div>
        <button class="dpad-btn" ontouchstart="dpadOn('LEFT',this)" ontouchend="dpadOff(this)" onmousedown="dpadOn('LEFT',this)" onmouseup="dpadOff(this)" onmouseleave="dpadOff(this)">LFT</button>
        <button class="dpad-btn" ontouchstart="dpadOn('STOP',this)" ontouchend="dpadOff(this)" onmousedown="dpadOn('STOP',this)" onmouseup="dpadOff(this)" onmouseleave="dpadOff(this)">STP</button>
        <button class="dpad-btn" ontouchstart="dpadOn('RIGHT',this)" ontouchend="dpadOff(this)" onmousedown="dpadOn('RIGHT',this)" onmouseup="dpadOff(this)" onmouseleave="dpadOff(this)">RGT</button>
        <div class="dpad-empty"></div>
        <button class="dpad-btn" ontouchstart="dpadOn('BACKWARD',this)" ontouchend="dpadOff(this)" onmousedown="dpadOn('BACKWARD',this)" onmouseup="dpadOff(this)" onmouseleave="dpadOff(this)">BCK</button>
        <div class="dpad-empty"></div>
      </div>
      <button class="conv-btn" id="convBtn" onclick="toggleRoller()">CONV ON</button>
      <button class="conv-btn" id="pumpBtnL" onclick="togglePump()" style="margin-top:2px">PUMP ON</button>
      <div class="pan-row">
        <span style="font-size:8px;letter-spacing:1px" class="section-label">PAN</span>
        <span class="pan-val" id="panValL">90°</span>
        <input type="range" class="horizontal" id="panSliderL" min="0" max="180" value="90" style="flex:1" oninput="onPan(this.value)">
        <button class="conv-btn" id="panCentreBtnL" onclick="setPanCentre()" style="padding:3px 6px;white-space:nowrap">CTR</button>
      </div>
      <div class="pan-row">
        <span style="font-size:8px;letter-spacing:1px" class="section-label">TILT</span>
        <span class="pan-val" id="tiltValL">90°</span>
        <input type="range" class="horizontal" id="tiltSliderL" min="0" max="180" value="90" style="flex:1" oninput="onTilt(this.value)">
        <button class="conv-btn" id="tiltCentreBtnL" onclick="setTiltCentre()" style="padding:3px 6px;white-space:nowrap">LVL</button>
      </div>
    </div>
  </div><!-- /left-panel -->
  <!-- Camera + portrait joystick -->
  <div class="right-panel">
    <div class="cam-section">
      <div class="cam-label">Camera</div>
      <img id="camPreview" src="" alt="">
      <div class="cam-status" id="camStatusText">Waiting...</div>
    </div>
    <div class="joy-section">
      <div class="joy-label">Joystick</div>
      <canvas id="joyCanvas"></canvas>
      <div class="joy-vals" id="joyVals">L: 0&nbsp;&nbsp;R: 0</div>
    </div>
  </div>
  <!-- Landscape joystick col 4 -->
  <div class="joy-land">
    <div class="joy-label">Joystick</div>
    <canvas id="joyCanvasL"></canvas>
    <div class="joy-vals" id="joyValsL">L: 0&nbsp;&nbsp;R: 0</div>
  </div>
  <!-- Conv+Pump vslider col 5 -->
  <div class="vslider-panel vslider-conv">
    <div class="vslider-wrap">
      <div class="vslider-label">Conv</div>
      <div class="vslider-val" id="rolValL">80</div>
      <input type="range" class="vertical" id="rolSliderL" min="0" max="100" value="80" oninput="onRollerPwm(this.value)">
    </div>
    <div class="vslider-wrap" style="margin-top:6px">
      <div class="vslider-label">Pump</div>
      <div class="vslider-val" id="pmpValL">80</div>
      <input type="range" class="vertical" id="pmpSliderL" min="0" max="100" value="80" oninput="onPumpPwm(this.value)">
    </div>
  </div>
</div><!-- /main -->
<footer>
  <span id="footerAddr">:5000</span>
  <span id="modeFooter">MANUAL</span>
</footer>
</div><!-- /layout -->
<script>
let theme='dark',rollerOn=false,pumpOn=false,mode='MANUAL',driveSpd=200,rolPwm=80,pmpPwm=80,panDeg=90,tiltDeg=90;

function toggleTheme(){
  theme=theme==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme',theme);
  document.getElementById('themeBtn').textContent=theme==='dark'?'LIGHT':'DARK';
  if(typeof joy!=='undefined')joy.draw();
  if(typeof joyL!=='undefined')joyL.draw();
}

function setMode(m){
  mode=m;
  const label=m==='AUTO'?'AUTONOMOUS':'MANUAL';
  [['tabManual','tabManualD'],['tabAuto','tabAutoD']].forEach(([a,b],i)=>{
    const active=i===(m==='MANUAL'?0:1);
    [a,b].forEach(id=>{const el=document.getElementById(id);if(el)el.classList.toggle('active',active);});
  });
  const md=document.getElementById('modeDisplay');if(md)md.textContent=label;
  const mf=document.getElementById('modeFooter'); if(mf)mf.textContent=label;
  updatePanTiltUI(m==='AUTO');
  post('/command',{cmd:m});
}

function updateTrack(el){
  if(!el)return;
  const mn=parseFloat(el.min)||0,mx=parseFloat(el.max)||100;
  const pct=((parseFloat(el.value)||0)-mn)/(mx-mn)*100;
  if(el.classList.contains('vertical'))
    el.style.background='linear-gradient(to bottom,#cc2222 '+(100-pct)+'%,#2a2a2a '+(100-pct)+'%)';
  else
    el.style.background='linear-gradient(to right,#cc2222 '+pct+'%,#2a2a2a '+pct+'%)';
}
function initTracks(){document.querySelectorAll('input[type=range]').forEach(el=>{updateTrack(el);el.addEventListener('input',()=>updateTrack(el));});}

function onSpeed(v){
  driveSpd=parseInt(v);
  ['spdVal','spdValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=v;});
  ['spdSlider','spdSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=v;updateTrack(e);}});
  post('/set_speed',{value:driveSpd});
}

function onRollerPwm(v){
  rolPwm=parseInt(v);
  ['rolVal','rolValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=v;});
  ['rolSlider','rolSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=v;updateTrack(e);}});
  post('/set_roller_pwm',{value:rolPwm});
}

function onPumpPwm(v){
  pmpPwm=parseInt(v);
  ['pmpVal','pmpValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=v;});
  ['pmpSlider','pmpSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=v;updateTrack(e);}});
  post('/set_pump_pwm',{value:pmpPwm});
}

function onPan(v){
  if(mode==='AUTO')return;
  panDeg=parseInt(v);
  ['panVal','panValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=panDeg+'°';});
  ['panSlider','panSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=panDeg;updateTrack(e);}});
  post('/set_pan',{value:panDeg});
}
function setPanCentre(){
  if(mode==='AUTO')return;
  panDeg=90;
  ['panVal','panValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent='90°';});
  ['panSlider','panSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=90;updateTrack(e);}});
  post('/set_pan',{value:90});
}

function onTilt(v){
  if(mode==='AUTO')return;
  tiltDeg=parseInt(v);
  ['tiltVal','tiltValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=tiltDeg+'°';});
  ['tiltSlider','tiltSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=tiltDeg;updateTrack(e);}});
  post('/set_tilt',{value:tiltDeg});
}
function setTiltCentre(){
  if(mode==='AUTO')return;
  tiltDeg=90;
  ['tiltVal','tiltValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent='90°';});
  ['tiltSlider','tiltSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=90;updateTrack(e);}});
  post('/set_tilt',{value:90});
}

function updatePanTiltUI(isAuto){
  ['panSlider','panSliderL'].forEach(id=>{const e=document.getElementById(id);if(e)e.disabled=isAuto;});
  ['tiltSlider','tiltSliderL'].forEach(id=>{const e=document.getElementById(id);if(e)e.disabled=isAuto;});
  ['panCentreBtn','panCentreBtnL'].forEach(id=>{const b=document.getElementById(id);if(b){b.disabled=isAuto;b.style.opacity=isAuto?'0.35':'1';}});
  ['tiltCentreBtn','tiltCentreBtnL'].forEach(id=>{const b=document.getElementById(id);if(b){b.disabled=isAuto;b.style.opacity=isAuto?'0.35':'1';}});
  const pNote=document.getElementById('panNote');if(pNote)pNote.textContent=isAuto?'AUTO: PID tracking — read-only':'MANUAL: drag to aim';
  const tNote=document.getElementById('tiltNote');if(tNote)tNote.textContent=isAuto?'AUTO: fixed during tracking':'MANUAL: drag to tilt camera';
}

function toggleRoller(){
  rollerOn=!rollerOn;
  [['rollerBtn',rollerOn?'CONVEYOR OFF':'CONVEYOR ON'],['convBtn',rollerOn?'CONV OFF':'CONV ON']].forEach(([id,txt])=>{
    const btn=document.getElementById(id);if(!btn)return;btn.textContent=txt;btn.classList.toggle('active',rollerOn);
  });
  post('/command',{cmd:rollerOn?'ROLLER_ON':'ROLLER_OFF'});
}

function togglePump(){
  pumpOn=!pumpOn;
  [['pumpBtn',pumpOn?'PUMP OFF':'PUMP ON'],['pumpBtnL',pumpOn?'PUMP OFF':'PUMP ON']].forEach(([id,txt])=>{
    const btn=document.getElementById(id);if(!btn)return;btn.textContent=txt;btn.classList.toggle('active',pumpOn);
  });
  post('/command',{cmd:pumpOn?'PUMP_ON':'PUMP_OFF'});
}

let dpadHeld=null;
function dpadOn(cmd,el){if(dpadHeld===cmd)return;dpadHeld=cmd;el.classList.add('held');post('/command',{cmd});}
function dpadOff(el){if(!dpadHeld)return;dpadHeld=null;el.classList.remove('held');post('/command',{cmd:'STOP'});}
function post(path,body){fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).catch(()=>{});}

function pollFrame(){
  const img=document.getElementById('camPreview');
  const tmp=new Image();
  tmp.onload=()=>{img.src=tmp.src;const s=document.getElementById('camStatusText');if(s)s.textContent='Live';};
  tmp.onerror=()=>{const s=document.getElementById('camStatusText');if(s)s.textContent='No stream';};
  tmp.src='/frame?'+Date.now();
}
setInterval(pollFrame,120);

function poll(){
  fetch('/status').then(r=>r.json()).then(d=>{
    const cd=document.getElementById('cmdDisplay');if(cd)cd.textContent=d.last_cmd||'---';
    const fd=document.getElementById('fpsDisplay');if(fd)fd.textContent=(d.fps||'0.0')+' fps';
    const ps=document.getElementById('piStatus');  if(ps)ps.textContent=d.pi_status||'---';
    const pd=document.getElementById('piDot');     if(pd)pd.className='dot '+(d.pi_ok?'ok':'err');
    const cd2=document.getElementById('camDot');   if(cd2)cd2.className='dot '+(d.cam_ok?'ok':'err');
    const os=document.getElementById('oilStatus');
    if(os){
      if(d.oil_tracking){
        const cov=d.oil_coverage_pct>0?` ${d.oil_coverage_pct.toFixed(1)}%`:'';
        os.textContent=`OIL:${d.oil_error>0?'+':''}${d.oil_error}px${cov}`;
      } else { os.textContent='OIL:SCAN'; }
    }
    if(d.roller_on!==rollerOn){
      rollerOn=d.roller_on;
      [['rollerBtn',rollerOn?'CONVEYOR OFF':'CONVEYOR ON'],['convBtn',rollerOn?'CONV OFF':'CONV ON']].forEach(([id,txt])=>{
        const btn=document.getElementById(id);if(!btn)return;btn.textContent=txt;btn.classList.toggle('active',rollerOn);
      });
    }
    if(d.pump_on!==undefined&&d.pump_on!==pumpOn){
      pumpOn=d.pump_on;
      [['pumpBtn',pumpOn?'PUMP OFF':'PUMP ON'],['pumpBtnL',pumpOn?'PUMP OFF':'PUMP ON']].forEach(([id,txt])=>{
        const btn=document.getElementById(id);if(!btn)return;btn.textContent=txt;btn.classList.toggle('active',pumpOn);
      });
    }
    if(d.pump_pwm!==undefined){
      const pa=document.activeElement?.id?.startsWith('pmp');
      if(!pa){
        ['pmpSlider','pmpSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=d.pump_pwm;updateTrack(e);}});
        ['pmpVal','pmpValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=d.pump_pwm;});
      }
    }
    if(d.mode!==mode)setMode(d.mode); else updatePanTiltUI(mode==='AUTO');
    if(d.pan_pos!==undefined){
      const sp=d.pan_pos;
      ['panVal','panValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=sp+'°';});
      const pa=document.activeElement?.id?.indexOf('pan')!==-1;
      if(mode==='AUTO'||!pa){panDeg=sp;['panSlider','panSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=sp;updateTrack(e);}});}
    }
    if(d.tilt_pos!==undefined){
      const st=d.tilt_pos;
      ['tiltVal','tiltValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=st+'°';});
      const ta=document.activeElement?.id?.indexOf('tilt')!==-1;
      if(!ta){tiltDeg=st;['tiltSlider','tiltSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=st;updateTrack(e);}});}
    }
    const sa=document.activeElement?.id?.startsWith('spd');
    if(!sa){
      ['spdSlider','spdSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=d.drive_speed;updateTrack(e);}});
      ['spdVal','spdValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=d.drive_speed;});
    }
    const ra=document.activeElement?.id?.startsWith('rol');
    if(!ra){
      ['rolSlider','rolSliderL'].forEach(id=>{const e=document.getElementById(id);if(e){e.value=d.roller_pwm;updateTrack(e);}});
      ['rolVal','rolValL'].forEach(id=>{const e=document.getElementById(id);if(e)e.textContent=d.roller_pwm;});
    }
  }).catch(()=>{});
}
setInterval(poll,700);

function makeJoystick(canvasEl,valsEl,onMove,onEnd){
  const ctx=canvasEl.getContext('2d');
  let CX,CY,OUTER_R,INNER_R,knobX,knobY,active=false,throttle=null;
  function resize(){
    const sec=canvasEl.parentElement;
    const s=Math.max(80,Math.min(sec.clientWidth-12,sec.clientHeight-40));
    canvasEl.width=s;canvasEl.height=s;
    CX=CY=s/2;OUTER_R=s/2-10;INNER_R=Math.max(14,s*0.13);
    knobX=CX;knobY=CY;draw();
  }
  function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
  function draw(){
    const acc=css('--accent')||'#cc2222',brd=css('--border')||'#222',srf=css('--surface')||'#111';
    ctx.clearRect(0,0,canvasEl.width,canvasEl.height);
    ctx.beginPath();ctx.arc(CX,CY,OUTER_R,0,Math.PI*2);
    ctx.strokeStyle=active?acc:brd;ctx.lineWidth=active?1.5:1;ctx.stroke();
    ctx.strokeStyle=brd;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(CX,CY-OUTER_R);ctx.lineTo(CX,CY+OUTER_R);ctx.stroke();
    ctx.beginPath();ctx.moveTo(CX-OUTER_R,CY);ctx.lineTo(CX+OUTER_R,CY);ctx.stroke();
    if(active){ctx.beginPath();ctx.moveTo(CX,CY);ctx.lineTo(knobX,knobY);ctx.strokeStyle=acc+'55';ctx.lineWidth=1;ctx.stroke();}
    ctx.beginPath();ctx.arc(knobX,knobY,INNER_R,0,Math.PI*2);
    ctx.fillStyle=active?acc+'22':srf;ctx.fill();ctx.strokeStyle=acc;ctx.lineWidth=active?2:1.5;ctx.stroke();
    ctx.beginPath();ctx.arc(CX,CY,2.5,0,Math.PI*2);ctx.fillStyle=brd;ctx.fill();
  }
  function tankMix(nx,ny){const md=driveSpd/255*100;let l=ny+nx,r=ny-nx;const mg=Math.max(Math.abs(l),Math.abs(r),1);return{l:Math.round(l/mg*md),r:Math.round(r/mg*md)};}
  function getPos(e){const rc=canvasEl.getBoundingClientRect(),sx=canvasEl.width/rc.width,sy=canvasEl.height/rc.height,src=e.touches?e.touches[0]:e;return{x:(src.clientX-rc.left)*sx,y:(src.clientY-rc.top)*sy};}
  function move(pos){let dx=pos.x-CX,dy=pos.y-CY,dist=Math.sqrt(dx*dx+dy*dy);if(dist>OUTER_R){dx=dx/dist*OUTER_R;dy=dy/dist*OUTER_R;}knobX=CX+dx;knobY=CY+dy;const nx=dx/OUTER_R,ny=-dy/OUTER_R,{l,r}=tankMix(nx,ny);if(valsEl)valsEl.innerHTML='L: '+l+'&nbsp;&nbsp;R: '+r;if(!throttle){throttle=setTimeout(()=>{onMove(+nx.toFixed(3),+ny.toFixed(3));throttle=null;},40);}}
  function start(e){e.preventDefault();active=true;move(getPos(e));draw();}
  function mv(e){e.preventDefault();if(!active)return;move(getPos(e));draw();}
  function end(e){e.preventDefault();active=false;knobX=CX;knobY=CY;if(valsEl)valsEl.innerHTML='L: 0&nbsp;&nbsp;R: 0';onEnd();draw();}
  canvasEl.addEventListener('touchstart',start,{passive:false});
  canvasEl.addEventListener('touchmove',mv,{passive:false});
  canvasEl.addEventListener('touchend',end,{passive:false});
  canvasEl.addEventListener('touchcancel',end,{passive:false});
  canvasEl.addEventListener('mousedown',start);
  canvasEl.addEventListener('mousemove',e=>{if(active)mv(e);});
  canvasEl.addEventListener('mouseup',end);
  canvasEl.addEventListener('mouseleave',e=>{if(active)end(e);});
  window.addEventListener('resize',()=>setTimeout(resize,60));
  resize();return{draw,resize};
}
const joy =makeJoystick(document.getElementById('joyCanvas'), document.getElementById('joyVals'), (x,y)=>post('/joystick',{x,y}),()=>post('/joystick_end',{}));
const joyL=makeJoystick(document.getElementById('joyCanvasL'),document.getElementById('joyValsL'),(x,y)=>post('/joystick',{x,y}),()=>post('/joystick_end',{}));
window.addEventListener('load',()=>{initTracks();setTimeout(()=>{joy.resize();joyL.resize();},100);});
</script>
</body>
</html>"""

HTML_BYTES = HTML_CONTENT.encode("utf-8")


class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._resp(200, "text/html; charset=utf-8", HTML_BYTES)
        elif p == "/frame":
            cam = cam_ref[0]
            if cam:
                with cam.frame_lock:
                    f = cam.frame.copy() if cam.frame is not None else None
                if f is not None:
                    _, jpg = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self._resp(200, "image/jpeg", jpg.tobytes()); return
            self._resp(503, "text/plain", b"no frame")
        elif p == "/status":
            body = json.dumps({
                "mode"            : S.get("mode"),
                "roller_on"       : S.get("roller_on"),
                "roller_pwm"      : S.get("roller_pwm"),
                "pump_on"         : S.get("pump_on"),
                "pump_pwm"        : S.get("pump_pwm"),
                "drive_speed"     : S.get("drive_speed"),
                "last_cmd"        : S.get("last_cmd"),
                "pi_ok"           : S.get("pi_ok"),
                "pi_status"       : S.get("pi_status"),
                "cam_ok"          : S.get("cam_ok"),
                "fps"             : round(S.get("fps"), 1),
                "oil_tracking"    : S.get("oil_tracking"),
                "oil_error"       : S.get("oil_error"),
                "oil_coverage_pct": round(S.get("oil_coverage_pct"), 1),
                "pan_pos"         : S.get("pan_pos"),
                "tilt_pos"        : S.get("tilt_pos"),
            }).encode()
            self._resp(200, "application/json", body)
        else:
            self._resp(404, "text/plain", b"not found")

    def do_POST(self):
        p  = urlparse(self.path).path
        ln = int(self.headers.get("Content-Length", 0))
        try:    data = json.loads(self.rfile.read(ln))
        except: data = {}

        if p == "/joystick":
            S.set("joy_x", float(data.get("x", 0)))
            S.set("joy_y", float(data.get("y", 0)))
            S.set("joy_active", True); self._ok("joy")
        elif p == "/joystick_end":
            S.set("joy_active", False); S.set("joy_x", 0.0); S.set("joy_y", 0.0)
            pi.cmd("/stop"); self._ok("joy_end")
        elif p == "/command":
            _web_cmd(data.get("cmd", "").upper()); self._ok("cmd")
        elif p == "/set_speed":
            v = int(data.get("value", 200)); S.set("drive_speed", v); pi.speed(v); self._ok("spd")
        elif p == "/set_roller_pwm":
            v = int(data.get("value", 80)); S.set("roller_pwm", v)
            if S.get("roller_on"): pi.roller_pwm(v)
            self._ok("rol")
        elif p == "/set_pump_pwm":
            v = int(data.get("value", 80)); S.set("pump_pwm", v)
            if S.get("pump_on"): pi.pump_pwm(v)
            self._ok("pmp")
        elif p == "/set_pan":
            v = max(0, min(180, int(data.get("value", 90))))
            S.set("pan_pos", v); pi.pan_set(v); self._ok("pan")
        elif p == "/set_tilt":
            v = max(0, min(180, int(data.get("value", 90))))
            S.set("tilt_pos", v); pi.tilt_set(v); self._ok("tilt")
        else:
            self._resp(404, "text/plain", b"not found")

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _resp(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", len(body))
        self._cors(); self.end_headers(); self.wfile.write(body)

    def _ok(self, msg):
        self._resp(200, "application/json",
                   json.dumps({"ok": True, "msg": msg}).encode())

    def log_message(self, *a): pass


def _web_cmd(cmd):
    ROUTES = {"FORWARD": "/forward", "BACKWARD": "/back",
              "LEFT": "/left", "RIGHT": "/right", "STOP": "/stop"}
    if cmd == "AUTO":
        S.set("mode", "AUTO"); S.set("last_cmd", "AUTO")
    elif cmd == "MANUAL":
        S.set("mode", "MANUAL"); pi.cmd("/stop"); S.set("last_cmd", "---")
    elif cmd == "ROLLER_ON":
        pi.roller_pwm(S.get("roller_pwm")); S.set("roller_on", True); S.set("last_cmd", "ROLLER ON")
    elif cmd == "ROLLER_OFF":
        pi.roller_pwm(0); S.set("roller_on", False); S.set("last_cmd", "ROLLER OFF")
    elif cmd == "PUMP_ON":
        pi.pump_pwm(S.get("pump_pwm")); S.set("pump_on", True); S.set("last_cmd", "PUMP ON")
    elif cmd == "PUMP_OFF":
        pi.pump_pwm(0); S.set("pump_on", False); S.set("last_cmd", "PUMP OFF")
    elif cmd in ROUTES and S.get("mode") == "MANUAL":
        pi.cmd(ROUTES[cmd]); S.set("last_cmd", cmd)


def start_web():
    srv = HTTPServer(("0.0.0.0", WEB_PORT), WebHandler)
    print(f"[WEB]  http://{PC_IP}:{WEB_PORT}")
    srv.serve_forever()


class App:
    DRIVE_STEP  = 10
    ROLLER_STEP = 5

    def __init__(self, root):
        self.root = root; self.running = True
        self.current_frame = None; self.frame_lock = threading.Lock()
        self.keys = set(); self.last_kc = None
        self.ai_sm = Smoother(CMD_COOLDOWN)
        root.title("Oil Spill Boat")
        root.configure(bg="#0a0a0a")
        root.protocol("WM_DELETE_WINDOW", self._close)
        self.cam = UDPCamera(); self.cam.start()
        cam_ref[0] = self.cam
        self._load_models()
        pi.start()
        self._build_ui()
        root.bind("<KeyPress>",   self._kp)
        root.bind("<KeyRelease>", self._kr)
        root.focus_set()
        threading.Thread(target=start_web,      daemon=True).start()
        threading.Thread(target=joystick_loop,  daemon=True).start()
        threading.Thread(target=self._vision,   daemon=True).start()
        threading.Thread(target=self._key_loop, daemon=True).start()
        self._tick()

    def _load_models(self):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpu_name = torch.cuda.get_device_name(0) if self._device == 'cuda' else 'No GPU'
        print(f"[DEVICE] {self._device.upper()}  {gpu_name}")
        if not os.path.exists(ENGINE_PATH):
            messagebox.showerror("Engine not found", ENGINE_PATH); sys.exit(1)
        print(f"[MODEL] Loading TensorRT engine...\n        {ENGINE_PATH}")
        self._is_trt = True
        import json as _json, zipfile as _zf
        _task = "detect"
        try:
            if _zf.is_zipfile(ENGINE_PATH):
                with _zf.ZipFile(ENGINE_PATH) as _z:
                    if "metadata.json" in _z.namelist():
                        _task = _json.loads(_z.read("metadata.json")).get("task", "detect")
        except Exception: pass
        if _task == "detect" and "seg" in ENGINE_PATH.lower(): _task = "segment"
        self._has_masks = (_task == "segment")
        print(f"[MODEL] Task: {_task}  |  Masks: {'YES' if self._has_masks else 'NO — box-only mode'}")
        self.model = YOLO(ENGINE_PATH, task=_task)
        print(f"[MODEL] Warming up ({OIL_WARMUP_RUNS} passes)...")
        dummy = np.zeros((OIL_IMG_SIZE, OIL_IMG_SIZE, 3), dtype=np.uint8)
        for _ in range(OIL_WARMUP_RUNS):
            self.model.predict(source=dummy, imgsz=OIL_IMG_SIZE, verbose=False,
                               stream=False, half=(self._device == 'cuda'))
        print("[MODEL] Ready")
        for i, n in self.model.names.items(): print(f"         {i}: {n}")

    def _build_ui(self):
        BG, PAN, ACC, DIM = "#0a0a0a", "#111111", "#ffffff", "#1c1c1c"
        MONO = ("Consolas", 9)
        self.video_lbl = tk.Label(self.root, bg="#000")
        self.video_lbl.grid(row=0, column=0, padx=(10, 5), pady=10, rowspan=20)
        rp = tk.Frame(self.root, bg=BG, width=250)
        rp.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        rp.grid_propagate(False)

        def sec(parent, title):
            f = tk.Frame(parent, bg=PAN, bd=0); f.pack(fill="x", pady=(0, 6))
            tk.Label(f, text=title, font=("Consolas", 8), bg=PAN, fg="#444",
                     anchor="w").pack(fill="x", padx=8, pady=(6, 2))
            tk.Frame(f, bg="#1e1e1e", height=1).pack(fill="x", padx=8)
            return f

        def labeled(parent, key, val):
            row = tk.Frame(parent, bg=PAN); row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=key, font=MONO, bg=PAN, fg="#555").pack(side="left")
            lbl = tk.Label(row, text=val, font=MONO, bg=PAN, fg=ACC, anchor="e")
            lbl.pack(side="right"); return lbl

        sf = sec(rp, "LAST COMMAND")
        self.cmd_lbl = tk.Label(sf, text="---", font=("Consolas", 16, "bold"), bg=PAN, fg=ACC)
        self.cmd_lbl.pack(padx=8, pady=(4, 8))

        stf = sec(rp, "STATUS")
        self.pi_lbl  = labeled(stf, "Pi",      "---")
        self.cam_lbl = labeled(stf, "Camera",  "---")
        self.fps_lbl = labeled(stf, "FPS",     "0.0")
        self.oil_lbl = labeled(stf, "Oil",     "SCAN")
        self.inf_lbl = labeled(stf, "InferMs", "--")
        tk.Frame(stf, bg=PAN, height=4).pack()

        mf = sec(rp, "MODE")
        mrow = tk.Frame(mf, bg=PAN); mrow.pack(padx=8, pady=(4, 8))
        self.mode_var = tk.StringVar(value="MANUAL")
        for txt, val in [("MANUAL", "MANUAL"), ("AUTO AI", "AUTO")]:
            tk.Radiobutton(mrow, text=txt, variable=self.mode_var, value=val,
                           bg=PAN, fg="#888", selectcolor=DIM, activebackground=PAN,
                           font=MONO, command=self._mode_changed).pack(side="left", padx=8)

        dsf = sec(rp, "DRIVE SPEED  (Up/Down)")
        drow = tk.Frame(dsf, bg=PAN); drow.pack(fill="x", padx=8, pady=(4, 8))
        self.spd_lbl = tk.Label(drow, text="200", width=4, font=("Consolas", 11, "bold"), bg=PAN, fg=ACC)
        self.spd_lbl.pack(side="left")
        self.spd_scale = tk.Scale(drow, from_=0, to=255, orient="horizontal", bg=BG, fg=ACC,
                                  troughcolor=DIM, highlightthickness=0, showvalue=False,
                                  command=self._spd_changed)
        self.spd_scale.set(200); self.spd_scale.pack(side="left", fill="x", expand=True)

        cf = sec(rp, "CONVEYOR  (L/R arrows + Space)")
        crow = tk.Frame(cf, bg=PAN); crow.pack(fill="x", padx=8, pady=(4, 4))
        self.rol_lbl = tk.Label(crow, text="80", width=4, font=("Consolas", 11, "bold"), bg=PAN, fg=ACC)
        self.rol_lbl.pack(side="left")
        self.rol_scale = tk.Scale(crow, from_=0, to=100, orient="horizontal", bg=BG, fg=ACC,
                                  troughcolor=DIM, highlightthickness=0, showvalue=False,
                                  command=self._rol_pwm_changed)
        self.rol_scale.set(80); self.rol_scale.pack(side="left", fill="x", expand=True)
        self.rol_btn = tk.Button(cf, text="ROLLER  ON", bg=DIM, fg=ACC,
                                 font=("Consolas", 9, "bold"), relief="flat", bd=0,
                                 activebackground="#222", command=self._toggle_roller)
        self.rol_btn.pack(padx=8, pady=(2, 8), fill="x")

        pmpf = sec(rp, "PUMP  (GPIO ENB=8  IN3=7  IN4=25)")
        pmprow = tk.Frame(pmpf, bg=PAN); pmprow.pack(fill="x", padx=8, pady=(4, 4))
        self.pmp_lbl = tk.Label(pmprow, text="80", width=4, font=("Consolas", 11, "bold"), bg=PAN, fg=ACC)
        self.pmp_lbl.pack(side="left")
        self.pmp_scale = tk.Scale(pmprow, from_=0, to=100, orient="horizontal", bg=BG, fg=ACC,
                                  troughcolor=DIM, highlightthickness=0, showvalue=False,
                                  command=self._pmp_pwm_changed)
        self.pmp_scale.set(80); self.pmp_scale.pack(side="left", fill="x", expand=True)
        self.pmp_btn = tk.Button(pmpf, text="PUMP  ON", bg=DIM, fg=ACC,
                                 font=("Consolas", 9, "bold"), relief="flat", bd=0,
                                 activebackground="#222", command=self._toggle_pump)
        self.pmp_btn.pack(padx=8, pady=(2, 8), fill="x")

        pf = sec(rp, "PAN SERVO  (0=left  90=centre  180=right)")
        prow = tk.Frame(pf, bg=PAN); prow.pack(fill="x", padx=8, pady=(4, 4))
        self.pan_lbl = tk.Label(prow, text="90", width=4, font=("Consolas", 11, "bold"), bg=PAN, fg=ACC)
        self.pan_lbl.pack(side="left")
        self.pan_scale = tk.Scale(prow, from_=0, to=180, orient="horizontal", bg=BG, fg=ACC,
                                  troughcolor=DIM, highlightthickness=0, showvalue=False,
                                  command=self._pan_changed)
        self.pan_scale.set(90); self.pan_scale.pack(side="left", fill="x", expand=True)
        self.pan_centre_btn = tk.Button(pf, text="CENTRE", bg=DIM, fg=ACC,
                                        font=("Consolas", 9, "bold"), relief="flat", bd=0,
                                        activebackground="#222", command=lambda: self._pan_set(90))
        self.pan_centre_btn.pack(padx=8, pady=(2, 8), fill="x")
        self.pan_mode_lbl = tk.Label(pf, text="MANUAL: drag to aim",
                                     font=("Consolas", 7), bg=PAN, fg="#555", anchor="w")
        self.pan_mode_lbl.pack(fill="x", padx=8, pady=(0, 6))

        tf = sec(rp, "TILT SERVO  (0=down  90=level  180=up)")
        trow = tk.Frame(tf, bg=PAN); trow.pack(fill="x", padx=8, pady=(4, 4))
        self.tilt_lbl = tk.Label(trow, text="90", width=4, font=("Consolas", 11, "bold"), bg=PAN, fg=ACC)
        self.tilt_lbl.pack(side="left")
        self.tilt_scale = tk.Scale(trow, from_=0, to=180, orient="horizontal", bg=BG, fg=ACC,
                                   troughcolor=DIM, highlightthickness=0, showvalue=False,
                                   command=self._tilt_changed)
        self.tilt_scale.set(90); self.tilt_scale.pack(side="left", fill="x", expand=True)
        self.tilt_centre_btn = tk.Button(tf, text="LEVEL", bg=DIM, fg=ACC,
                                         font=("Consolas", 9, "bold"), relief="flat", bd=0,
                                         activebackground="#222", command=lambda: self._tilt_set(90))
        self.tilt_centre_btn.pack(padx=8, pady=(2, 8), fill="x")
        self.tilt_mode_lbl = tk.Label(tf, text="MANUAL: drag to tilt",
                                      font=("Consolas", 7), bg=PAN, fg="#555", anchor="w")
        self.tilt_mode_lbl.pack(fill="x", padx=8, pady=(0, 6))

        wf = sec(rp, "WEB INTERFACE")
        tk.Label(wf, text=f"http://{PC_IP}:{WEB_PORT}",
                 font=("Consolas", 8), bg=PAN, fg="#555").pack(padx=8, pady=(4, 8))
        tk.Button(rp, text="QUIT & STOP", bg="#1a0000", fg="#cc3333",
                  font=("Consolas", 9, "bold"), relief="flat", bd=0,
                  activebackground="#220000", command=self._close).pack(fill="x", pady=(4, 0))

    def _kp(self, e):
        k = e.keysym.lower(); self.keys.add(k)
        if k == "space": self._toggle_roller()
        elif k == "up":
            v = min(255, S.get("drive_speed") + self.DRIVE_STEP)
            S.set("drive_speed", v); self.spd_scale.set(v); pi.speed(v)
        elif k == "down":
            v = max(0, S.get("drive_speed") - self.DRIVE_STEP)
            S.set("drive_speed", v); self.spd_scale.set(v); pi.speed(v)
        elif k == "left":
            v = max(0, S.get("roller_pwm") - self.ROLLER_STEP)
            S.set("roller_pwm", v); self.rol_scale.set(v)
            if S.get("roller_on"): pi.roller_pwm(v)
        elif k == "right":
            v = min(100, S.get("roller_pwm") + self.ROLLER_STEP)
            S.set("roller_pwm", v); self.rol_scale.set(v)
            if S.get("roller_on"): pi.roller_pwm(v)

    def _kr(self, e): self.keys.discard(e.keysym.lower())

    def _key_loop(self):
        while self.running:
            if S.get("mode") == "MANUAL" and not S.get("joy_active"):
                k = self.keys; spd = int(S.get("drive_speed") / 255 * 100)
                w = "w" in k; s = "s" in k; a = "a" in k; d = "d" in k
                if   w and a: l, r = spd, 0
                elif w and d: l, r = 0, spd
                elif w:       l, r = spd, spd
                elif s and a: l, r = -spd, 0
                elif s and d: l, r = 0, -spd
                elif s:       l, r = -spd, -spd
                elif a:       l, r = -spd, spd
                elif d:       l, r = spd, -spd
                else:         l, r = 0, 0
                if (l, r) != self.last_kc:
                    self.last_kc = (l, r)
                    if l == 0 and r == 0: pi.cmd("/stop"); S.set("last_cmd", "STOP")
                    else: pi.pwm(l, r); S.set("last_cmd", f"L{l:+d} R{r:+d}")
            time.sleep(0.04)

    def _vision(self):
        fc = 0; ft = time.time()
        servo_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fps_buf    = []
        fx_history = []
        last_send  = 0.0
        was_auto   = False

        while self.running:
            ret, frame = self.cam.read()
            if not ret or frame is None: time.sleep(0.05); continue

            now  = time.time()
            mode = S.get("mode")
            cx   = FRAME_W // 2
            oil_info = None
            dets     = []

            if mode == "AUTO":
                was_auto = True
                total_px = FRAME_W * FRAME_H
                _t0      = time.perf_counter()
                result   = self.model.predict(
                    source=frame, imgsz=OIL_IMG_SIZE, conf=OIL_CONF, iou=OIL_IOU,
                    max_det=20, verbose=False, stream=False,
                    half=(self._device == 'cuda'))[0]
                _inf_ms = (time.perf_counter() - _t0) * 1000
                fps_buf.append(1.0 / (_inf_ms / 1000 + 1e-9))
                if len(fps_buf) > 30: fps_buf.pop(0)
                S.set("oil_infer_ms", _inf_ms)

                boxes = result.boxes
                masks = result.masks if self._has_masks else None
                best_cx = best_cy = None
                best_area_pct = total_oil_pct = 0.0
                draw_items = []

                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        cid          = int(boxes.cls[i].item())
                        conf_val     = float(boxes.conf[i].item())
                        x1,y1,x2,y2 = map(int, boxes.xyxy[i].tolist())
                        color        = OIL_CLASS_COLORS_BGR.get(cid, OIL_DEFAULT_COLOR)
                        name         = self.model.names.get(cid, f"cls{cid}")
                        mask_full = m_cx = m_cy = None
                        area_pct = 0.0
                        if masks is not None and i < len(masks.data):
                            mask_np   = masks.data[i].cpu().numpy()
                            mask_full = cv2.resize(mask_np, (FRAME_W, FRAME_H),
                                                   interpolation=cv2.INTER_NEAREST) > 0.5
                            area_pct  = mask_full.sum() / total_px
                            ys, xs    = np.where(mask_full)
                            if len(xs): m_cx = int(xs.mean()); m_cy = int(ys.mean())
                        else:
                            area_pct = ((x2-x1)*(y2-y1)) / total_px
                            m_cx = (x1+x2)//2; m_cy = (y1+y2)//2
                        valid = _is_oil(self.model, cid) and area_pct >= OIL_MIN_AREA_FRAC
                        if valid:
                            total_oil_pct += area_pct * 100
                            if area_pct > best_area_pct:
                                best_area_pct = area_pct; best_cx = m_cx; best_cy = m_cy
                        draw_items.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"color":color,
                                           "label":f"{name} {conf_val:.2f}","mask_full":mask_full,
                                           "cx":m_cx,"cy":m_cy,"valid":valid,"area_pct":area_pct})
                        if valid:
                            dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                                         "cx":m_cx if m_cx is not None else (x1+x2)//2,
                                         "cy":m_cy if m_cy is not None else (y1+y2)//2,
                                         "area":int(area_pct*total_px),"conf":conf_val,"label":name})

                S.set("oil_coverage_pct", total_oil_pct)

                if best_cx is not None:
                    fx_history.append(best_cx)
                    if len(fx_history) > 3: fx_history.pop(0)
                    fx_avg = int(sum(fx_history) / len(fx_history))
                    dx     = cx - fx_avg
                    S.set("oil_tracking", True); S.set("oil_error", dx)
                    if abs(dx) > OIL_DEAD_ZONE and (now - last_send) >= OIL_SEND_RATE:
                        servo_sock.sendto(str(dx).encode(), (PI_IP, UDP_SERVO_PORT))
                        last_send = now
                    est_pan = max(0, min(180, 90 - int(dx / (FRAME_W / 2) * 90)))
                    S.set("pan_pos", est_pan)
                    oil_info = {"draw_items":draw_items,"fx_avg":fx_avg,"best_cy":best_cy,
                                "dx":dx,"total_oil_pct":total_oil_pct,"tracking":True}
                else:
                    fx_history.clear()
                    S.set("oil_tracking", False); S.set("oil_error", 0)
                    if (now - last_send) >= OIL_SEND_RATE:
                        servo_sock.sendto(b"NOSPILL", (PI_IP, UDP_SERVO_PORT))
                        last_send = now
                    oil_info = {"draw_items":draw_items,"fx_avg":None,"best_cy":None,
                                "dx":0,"total_oil_pct":total_oil_pct,"tracking":False}
            else:
                if was_auto:
                    servo_sock.sendto(b"NOSPILL", (PI_IP, UDP_SERVO_PORT)); was_auto = False
                fx_history.clear()
                S.set("oil_tracking", False); S.set("oil_error", 0); S.set("oil_coverage_pct", 0.0)

            ann = self._draw(frame.copy(), dets, oil_info)
            if mode == "AUTO": self._ai(decide(dets))
            with self.frame_lock: self.current_frame = ann
            fc += 1
            if time.time() - ft >= 1.0:
                S.set("fps", fc / (time.time() - ft)); fc = 0; ft = time.time()

    def _draw(self, frame, dets, oil_info=None):
        BOX_OIL = (0, 200, 50); BOX_CLOSE = (0, 50, 255); TEXT_CLR = (255, 255, 255)
        cx = FRAME_W // 2
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (LEFT_ZONE_END, FRAME_H), (0, 15, 5), -1)
        cv2.rectangle(ov, (RIGHT_ZONE_START, 0), (FRAME_W, FRAME_H), (0, 15, 5), -1)
        cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
        cv2.line(frame, (LEFT_ZONE_END, 0), (LEFT_ZONE_END, FRAME_H), (50, 80, 60), 1)
        cv2.line(frame, (RIGHT_ZONE_START, 0), (RIGHT_ZONE_START, FRAME_H), (50, 80, 60), 1)
        cv2.putText(frame, "LEFT",   (4, 16),               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 50),  1)
        cv2.putText(frame, "CENTER", (FRAME_W//2-32, 16),   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "RIGHT",  (RIGHT_ZONE_START+2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 180, 255), 1)

        for d in dets:
            close = d["area"] / (FRAME_W * FRAME_H) >= PROXIMITY_RATIO
            col   = BOX_CLOSE if close else BOX_OIL
            z     = "L" if d["cx"] < LEFT_ZONE_END else "R" if d["cx"] > RIGHT_ZONE_START else "C"
            tag   = f"{d['label']} {d['conf']:.2f}[{z}]" + (" COLLECT" if close else "")
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), col, 2)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (d["x1"], d["y1"]-th-6), (d["x1"]+tw+3, d["y1"]), col, -1)
            cv2.putText(frame, tag, (d["x1"]+2, d["y1"]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_CLR, 1)
            cv2.circle(frame, (d["cx"], d["cy"]), 4, col, -1)

        if oil_info is not None:
            total_oil_pct = oil_info["total_oil_pct"]
            for item in oil_info["draw_items"]:
                color = item["color"]; x1,y1 = item["x1"],item["y1"]; x2,y2 = item["x2"],item["y2"]
                mask_full = item["mask_full"]; m_cx = item["cx"]; m_cy = item["cy"]
                valid = item["valid"]; area_pct = item["area_pct"]
                if not valid:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)
                    cv2.putText(frame, f"{item['label']} skip", (x1, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (60, 60, 60), 1); continue
                if mask_full is not None:
                    overlay = frame.copy(); overlay[mask_full] = color
                    cv2.addWeighted(overlay, OIL_MASK_ALPHA, frame, 1 - OIL_MASK_ALPHA, 0, frame)
                    mask_u8 = (mask_full * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                lbl = f"{item['label']} {area_pct*100:.1f}%"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(frame, (x1, y1-lh-6), (x1+lw+3, y1), color, -1)
                cv2.putText(frame, lbl, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,255,255), 1)
                if m_cx is not None and m_cy is not None:
                    cv2.drawMarker(frame, (m_cx, m_cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)

            cv2.line(frame, (cx, 0), (cx, FRAME_H), (255, 100, 0), 1)
            if oil_info["tracking"] and oil_info["fx_avg"] is not None:
                fx_avg  = oil_info["fx_avg"]; best_cy = oil_info["best_cy"] or (FRAME_H // 2)
                dx      = oil_info["dx"]
                cv2.line(frame, (cx, best_cy), (fx_avg, best_cy), (0, 60, 255), 1)
                cv2.circle(frame, (fx_avg, best_cy), 7, (0, 60, 255), -1)
                cv2.putText(frame, f"err:{dx:+d}px", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1)
                cv2.putText(frame, "TARGET", (fx_avg-24, best_cy-12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,60,255), 1)
            else:
                cv2.putText(frame, "SCANNING", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100,100,255), 1)

            if total_oil_pct > 0:
                alert_col = (0, 0, 220) if total_oil_pct > 10 else (0, 120, 220)
                cv2.rectangle(frame, (0, FRAME_H-30), (FRAME_W, FRAME_H), alert_col, -1)
                cv2.putText(frame, f"  OIL DETECTED — {total_oil_pct:.1f}% coverage",
                            (6, FRAME_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "OIL TRACKER PAUSED", (8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        mode     = S.get("mode")
        mode_col = (0, 200, 80) if mode == "AUTO" else (180, 180, 180)
        fps_val  = S.get("fps")
        fps_color = (0,220,0) if fps_val >= 25 else (0,220,255) if fps_val >= 15 else (0,60,255)
        hud_y = FRAME_H-36 if (oil_info is not None and oil_info.get("total_oil_pct",0)>0) else FRAME_H-8
        cv2.putText(frame, mode,              (8, hud_y),          cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_col,  1)
        cv2.putText(frame, f"FPS {fps_val:.1f}", (FRAME_W-72,hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fps_color, 1)
        if mode == "AUTO":
            inf_ms = S.get("oil_infer_ms")
            cv2.putText(frame, f"OIL {inf_ms:.0f}ms", (FRAME_W-90, hud_y-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,0), 1)
        return frame

    def _ai(self, cmd):
        if cmd == "COLLECT":
            if self.ai_sm.should("STOP"): pi.cmd("/stop"); S.set("last_cmd", "STOP")
            if not S.get("roller_on"):
                pi.roller_pwm(S.get("roller_pwm")); S.set("roller_on", True); S.set("last_cmd", "COLLECT")
        else:
            if S.get("roller_on") and cmd != "STOP": pi.roller_pwm(0); S.set("roller_on", False)
            if self.ai_sm.should(cmd):
                RMAP = {"FORWARD":"/forward","BACKWARD":"/back","LEFT":"/left","RIGHT":"/right","STOP":"/stop"}
                pi.cmd(RMAP.get(cmd, "/stop")); S.set("last_cmd", cmd)

    def _toggle_roller(self):
        if S.get("roller_on"):
            pi.roller_pwm(0); S.set("roller_on", False); S.set("last_cmd", "ROLLER OFF")
        else:
            pi.roller_pwm(S.get("roller_pwm")); S.set("roller_on", True); S.set("last_cmd", "ROLLER ON")

    def _toggle_pump(self):
        if S.get("pump_on"):
            pi.pump_pwm(0); S.set("pump_on", False); S.set("last_cmd", "PUMP OFF")
        else:
            pi.pump_pwm(S.get("pump_pwm")); S.set("pump_on", True); S.set("last_cmd", "PUMP ON")

    def _spd_changed(self, v):
        iv = int(v); S.set("drive_speed", iv); pi.speed(iv)

    def _rol_pwm_changed(self, v):
        iv = int(v); S.set("roller_pwm", iv)
        if S.get("roller_on"): pi.roller_pwm(iv)

    def _pmp_pwm_changed(self, v):
        iv = int(v); S.set("pump_pwm", iv)
        if S.get("pump_on"): pi.pump_pwm(iv)

    def _pan_changed(self, v):
        iv = int(v); S.set("pan_pos", iv); self.pan_lbl.config(text=str(iv))
        if S.get("mode") == "MANUAL": pi.pan_set(iv)

    def _pan_set(self, v):
        S.set("pan_pos", v); self.pan_scale.set(v); self.pan_lbl.config(text=str(v))
        if S.get("mode") == "MANUAL": pi.pan_set(v)

    def _tilt_changed(self, v):
        iv = int(v); S.set("tilt_pos", iv); self.tilt_lbl.config(text=str(iv))
        if S.get("mode") == "MANUAL": pi.tilt_set(iv)

    def _tilt_set(self, v):
        S.set("tilt_pos", v); self.tilt_scale.set(v); self.tilt_lbl.config(text=str(v))
        if S.get("mode") == "MANUAL": pi.tilt_set(v)

    def _mode_changed(self):
        m = self.mode_var.get(); S.set("mode", m)
        if m == "MANUAL": pi.cmd("/stop"); S.set("last_cmd", "---")

    def _tick(self):
        if not self.running: return
        with self.frame_lock: f = self.current_frame
        if f is not None:
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            self.video_lbl.configure(image=img); self.video_lbl.image = img

        self.cmd_lbl.config(text=S.get("last_cmd"))
        self.spd_lbl.config(text=str(S.get("drive_speed")))
        self.rol_lbl.config(text=str(S.get("roller_pwm")))
        self.fps_lbl.config(text=f"{S.get('fps'):.1f}")

        pi_ok = S.get("pi_ok")
        self.pi_lbl.config(text=S.get("pi_status"), fg="#fff" if pi_ok else "#cc3333")
        cam_ok = S.get("cam_ok")
        self.cam_lbl.config(text="OK" if cam_ok else "Waiting", fg="#fff" if cam_ok else "#555")

        cov = S.get("oil_coverage_pct")
        oil_txt = (f"err {S.get('oil_error'):+d}  {cov:.1f}%" if cov > 0
                   else f"err {S.get('oil_error'):+d}") if S.get("oil_tracking") else "SCAN"
        self.oil_lbl.config(text=oil_txt)
        self.inf_lbl.config(text=f"{S.get('oil_infer_ms'):.0f}ms")

        roller = S.get("roller_on")
        self.rol_btn.config(text="ROLLER OFF" if roller else "ROLLER  ON",
                            fg="#cc3333" if roller else "#fff")
        pump = S.get("pump_on")
        self.pmp_lbl.config(text=str(S.get("pump_pwm")))
        self.pmp_btn.config(text="PUMP  OFF" if pump else "PUMP  ON",
                            fg="#cc3333" if pump else "#fff")

        mode = S.get("mode")
        ACC  = "#ffffff"
        self.mode_var.set(mode)

        pan_v = S.get("pan_pos")
        self.pan_lbl.config(text=str(pan_v))
        self.pan_scale.config(state="disabled" if mode == "AUTO" else "normal")
        self.pan_scale.set(pan_v)
        self.pan_centre_btn.config(state="disabled" if mode == "AUTO" else "normal",
                                   fg="#555" if mode == "AUTO" else ACC)
        self.pan_mode_lbl.config(
            text="AUTO: PID tracking — slider read-only" if mode == "AUTO"
                 else "MANUAL: drag to aim  |  patrol in AUTO")

        tilt_v = S.get("tilt_pos")
        self.tilt_lbl.config(text=str(tilt_v))
        self.tilt_scale.config(state="disabled" if mode == "AUTO" else "normal")
        self.tilt_scale.set(tilt_v)
        self.tilt_centre_btn.config(state="disabled" if mode == "AUTO" else "normal",
                                    fg="#555" if mode == "AUTO" else ACC)
        self.tilt_mode_lbl.config(
            text="AUTO: fixed during tracking" if mode == "AUTO"
                 else "MANUAL: drag to tilt camera")

        self.root.after(100, self._tick)

    def _close(self):
        self.running = False; pi.stop_thread(); self.cam.stop()
        time.sleep(0.4); self.root.destroy()


def main():
    root = tk.Tk(); root.resizable(False, False)
    App(root); root.mainloop()


if __name__ == "__main__":
    main()