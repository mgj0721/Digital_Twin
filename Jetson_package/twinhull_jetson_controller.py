#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TwinHull Jetson Nano 통합 제어 프로그램
#
# 기존 detect_working_websocket.py + web_controller.py 통합본
#
# 유지되는 구조:
#   Jetson
#   ├── IMX219 Camera -> TensorRT YOLO -> Detection
#   │                                      └-> WebSocket Client -> Laptop:5000
#   │
#   └── Flask:5000 -> Browser -> Arduino Serial /dev/ttyUSB0
#
# WebSocket:
#   Jetson -> Laptop
#   IP   : <LAPTOP_LAN_IP>   # TODO: 아래 LAPTOP_IP 와 동일한 값으로 채우세요 (예: 192.168.0.25)
#   Port : 5000
#
# Flask:
#   Jetson 0.0.0.0:5000
#
# Arduino:
#   /dev/ttyUSB0 @ 115200

import time
import ctypes
import json
import asyncio
import threading
import queue

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import websockets

from flask import Flask, render_template_string
import serial


# =====================================================
# Configuration - Original Values Preserved
# =====================================================

# TODO: TensorRT 엔진(.engine) 파일의 실제 절대경로로 교체하세요.
#       (예: "/home/사용자명/model/best.engine")
ENGINE_PATH = "/home/<YOUR_USERNAME>/model/best.engine"

# Jetson -> Laptop WebSocket
# TODO: 노트북(중계 서버)이 접속된 실제 LAN IP 주소로 교체하세요. (예: "192.168.0.25")
LAPTOP_IP = "<LAPTOP_LAN_IP>"
WEBSOCKET_PORT = 5000
WEBSOCKET_URL = "ws://{}:{}".format(LAPTOP_IP, WEBSOCKET_PORT)
WS_RECONNECT_DELAY = 3.0

# Camera
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# TensorRT / YOLO
INPUT_W = 320
INPUT_H = 320

CONF_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45

# [cx, cy, w, h, confidence, class_probability]
OUTPUT_FORMAT = "cxcywh"

# Arduino
ARDUINO_PORT = "/dev/ttyUSB0"
ARDUINO_BAUDRATE = 115200

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =====================================================
# ArUco -> Surface Mapping
# =====================================================
# 현재 Unity에서 사용하는 surface ID와 동일하게 맞춘다.
#
# Front:
#   ArUco 0,1,2,3
# Back:
#   ArUco 4,5,6,7
#
# 각 면에서 4개의 마커 중심을 다음 순서로 배치한다고 가정:
#   0.0,0.0 = 좌상
#   1.0,0.0 = 우상
#   1.0,1.0 = 우하
#   0.0,1.0 = 좌하
#
# 실제 마커 배치가 다르면 아래 ID 매핑만 수정하면 된다.

SURFACE_MARKER_LAYOUT = {
    "foam-board-front": {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
    },
    "foam-board-back": {
        4: (0.0, 0.0),
        5: (1.0, 0.0),
        6: (1.0, 1.0),
        7: (0.0, 1.0),
    },
}

MIN_SURFACE_MARKERS = 4


# =====================================================
# GStreamer
# =====================================================

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# =====================================================
# TensorRT Engine
# =====================================================

class TensorRTEngine:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError("TensorRT engine을 로드하지 못했습니다.")

        self.context = self.engine.create_execution_context()

        self.input_index = None
        self.output_index = None

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)

            print(
                "Binding {}: name={}, shape={}, dtype={}".format(
                    i, name, shape, dtype
                )
            )

            if self.engine.binding_is_input(i):
                self.input_index = i
            else:
                self.output_index = i

        if self.input_index is None or self.output_index is None:
            raise RuntimeError("입력/출력 binding을 찾지 못했습니다.")

        self.input_shape = tuple(
            self.engine.get_binding_shape(self.input_index)
        )
        self.output_shape = tuple(
            self.engine.get_binding_shape(self.output_index)
        )

        if self.input_shape != (1, 3, INPUT_H, INPUT_W):
            print(
                "주의: 예상 입력 (1,3,320,320), 실제 입력:",
                self.input_shape
            )

        print("Input shape :", self.input_shape)
        print("Output shape:", self.output_shape)

        input_size = trt.volume(self.input_shape)
        output_size = trt.volume(self.output_shape)

        self.input_host = cuda.pagelocked_empty(
            input_size, np.float32
        )
        self.output_host = cuda.pagelocked_empty(
            output_size, np.float32
        )

        self.input_device = cuda.mem_alloc(
            self.input_host.nbytes
        )
        self.output_device = cuda.mem_alloc(
            self.output_host.nbytes
        )

        self.bindings = [0] * self.engine.num_bindings

        self.bindings[self.input_index] = int(
            self.input_device
        )
        self.bindings[self.output_index] = int(
            self.output_device
        )

        self.stream = cuda.Stream()
        self.debug_printed = False

    def infer(self, image):
        # BGR uint8 -> RGB float32 -> [0,1] -> CHW
        resized = cv2.resize(
            image,
            (INPUT_W, INPUT_H),
            interpolation=cv2.INTER_LINEAR
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.ascontiguousarray(tensor)

        np.copyto(
            self.input_host,
            tensor.ravel()
        )

        cuda.memcpy_htod_async(
            self.input_device,
            self.input_host,
            self.stream
        )

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        cuda.memcpy_dtoh_async(
            self.output_host,
            self.output_device,
            self.stream
        )

        self.stream.synchronize()

        raw = self.output_host.reshape(-1, 6)

        print(
            "RAW TOP:",
            raw[np.argmax(raw[:, 4])]
        )

        return self.output_host.copy().reshape(
            self.output_shape
        )


# =====================================================
# WebSocket Client
# Original Jetson -> Laptop structure preserved
# =====================================================

class WebSocketClient:
    """
    Jetson WebSocket Client.

    Sends detection_frame JSON to the laptop WebSocket server.
    """

    def __init__(self, url):
        self.url = url
        self.websocket = None
        self.sequence = 0
        self.connected = False

        self.stop_event = threading.Event()
        self.message_queue = queue.Queue(maxsize=5)

        self.thread = threading.Thread(
            target=self._thread_main
        )
        self.thread.daemon = True
        self.thread.start()

        while (
            not self.connected
            and not self.stop_event.is_set()
        ):
            time.sleep(0.05)

    def _thread_main(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(
                self._connection_loop()
            )

        except Exception as e:
            print(
                "WebSocket thread 오류:",
                e
            )

        finally:
            try:
                loop.close()
            except Exception:
                pass

    async def _connection_loop(self):
        while not self.stop_event.is_set():

            try:
                print(
                    "WebSocket 연결 시도:",
                    self.url
                )

                async with websockets.connect(
                    self.url
                ) as websocket:

                    self.websocket = websocket
                    self.connected = True

                    print(
                        "노트북 서버 WebSocket 연결 성공"
                    )

                    await websocket.send("JETSON")

                    print(
                        "JETSON 등록 완료"
                    )

                    while not self.stop_event.is_set():

                        try:
                            message = (
                                self.message_queue.get(
                                    timeout=0.1
                                )
                            )

                        except queue.Empty:
                            await asyncio.sleep(
                                0.001
                            )
                            continue

                        try:
                            await websocket.send(
                                message
                            )

                        except Exception as e:
                            print(
                                "WebSocket 전송 오류:",
                                e
                            )

                            try:
                                self.message_queue.put_nowait(
                                    message
                                )
                            except queue.Full:
                                pass

                            raise

            except Exception as e:

                if not self.stop_event.is_set():

                    print(
                        "WebSocket 연결 오류:",
                        e
                    )

                    print(
                        "{}초 후 재연결...".format(
                            WS_RECONNECT_DELAY
                        )
                    )

                    self.connected = False
                    self.websocket = None

                    await asyncio.sleep(
                        WS_RECONNECT_DELAY
                    )

            finally:
                self.connected = False
                self.websocket = None

    def send_detection_frame(
        self,
        detections,
        camera_width,
        camera_height,
        surface_context=None
    ):
        """
        YOLO detection을 detection_frame으로 전송한다.

        surface_context:
            {
                "id": "foam-board-front" 또는 "foam-board-back",
                "homography": 3x3 numpy array
            }

        YOLO bbox 중심점을 homography로 [0,1] x [0,1] 좌표로
        변환하고 surface.id/u/v를 함께 전송한다.
        """

        frame = {
            "type": "detection_frame",
            "schemaVersion": 1,
            "source": "yolov5",

            # Unity Receiver 호환:
            # 기존 sequence를 유지하면서 seq도 함께 보낸다.
            "sequence": self.sequence,
            "seq": self.sequence,

            "sentAtUnixMs": int(
                time.time() * 1000
            ),

            "camera": {
                "width": int(camera_width),
                "height": int(camera_height)
            },

            "detections": []
        }

        for index, detection in enumerate(
            detections
        ):
            x1, y1, x2, y2, confidence = detection

            bbox = {
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            }

            item = {
                "id": "det-{}".format(index),

                # 현재 TensorRT 모델의 실제 클래스
                "label": "crack",
                "className": "crack",

                "confidence": float(confidence),

                "bbox": bbox
            }

            # -------------------------------------------------
            # YOLO bbox 중심 -> surface u/v
            # -------------------------------------------------

            if (
                surface_context is not None and
                surface_context.get("homography") is not None and
                surface_context.get("id") is not None
            ):
                try:
                    cx = (
                        bbox["x"] +
                        bbox["width"] / 2.0
                    )

                    cy = (
                        bbox["y"] +
                        bbox["height"] / 2.0
                    )

                    point = np.array(
                        [
                            [
                                [cx, cy]
                            ]
                        ],
                        dtype=np.float32
                    )

                    mapped = cv2.perspectiveTransform(
                        point,
                        surface_context["homography"]
                    )

                    u = float(
                        mapped[0][0][0]
                    )

                    v = float(
                        mapped[0][0][1]
                    )

                    # 카메라 밖/면 밖으로 크게 벗어난 검출은
                    # surface를 붙이지 않는다.
                    if (
                        -0.05 <= u <= 1.05 and
                        -0.05 <= v <= 1.05
                    ):
                        item["surface"] = {
                            "id": surface_context["id"],
                            "u": round(
                                max(0.0, min(1.0, u)),
                                4
                            ),
                            "v": round(
                                max(0.0, min(1.0, v)),
                                4
                            )
                        }

                except Exception as e:
                    print(
                        "Surface mapping 오류:",
                        e
                    )

            frame["detections"].append(item)

        message = json.dumps(
            frame,
            separators=(",", ":")
        )

        if not self.connected:
            return

        try:
            self.message_queue.put_nowait(
                message
            )
            self.sequence += 1

        except queue.Full:

            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.message_queue.put_nowait(
                    message
                )
                self.sequence += 1
            except queue.Full:
                pass

    def close(self):
        self.stop_event.set()
        self.connected = False
        self.websocket = None

        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


# =====================================================
# YOLO Postprocess
# Original logic preserved
# =====================================================

def postprocess(
    output,
    original_w,
    original_h
):
    """
    Expected output:
        (1, 6300, 6)

    Each row:
        [cx, cy, w, h, confidence, class_id]
    """

    pred = output.reshape(-1, 6)

    boxes = []
    scores = []

    scale_x = (
        float(original_w) / INPUT_W
    )
    scale_y = (
        float(original_h) / INPUT_H
    )

    for row in pred:

        a, b, c, d, obj, cls_prob = [
            float(v) for v in row
        ]

        # Single-class model:
        # confidence = objectness * class probability
        conf = obj * cls_prob

        if conf < CONF_THRESHOLD:
            continue

        if OUTPUT_FORMAT == "xyxy":

            x1, y1, x2, y2 = (
                a, b, c, d
            )

        else:

            # cx, cy, w, h -> x1, y1, x2, y2
            x1 = a - c / 2.0
            y1 = b - d / 2.0
            x2 = a + c / 2.0
            y2 = b + d / 2.0

        # Clamp in model-image coordinates
        x1 = max(
            0.0,
            min(float(INPUT_W), x1)
        )
        y1 = max(
            0.0,
            min(float(INPUT_H), y1)
        )
        x2 = max(
            0.0,
            min(float(INPUT_W), x2)
        )
        y2 = max(
            0.0,
            min(float(INPUT_H), y2)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # Scale to original camera frame
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        scores.append(conf)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    results = []

    if len(indices) > 0:

        for idx in np.array(
            indices
        ).reshape(-1):

            x, y, w, h = boxes[
                int(idx)
            ]

            results.append(
                (
                    x,
                    y,
                    x + w,
                    y + h,
                    scores[int(idx)]
                )
            )

    return results


# =====================================================
# Arduino Serial
# Original web_controller.py logic preserved
# =====================================================

arduino = None
arduino_lock = threading.Lock()


def init_arduino():
    global arduino

    print("========================================")
    print("Arduino Serial")
    print("Port     :", ARDUINO_PORT)
    print("Baudrate :", ARDUINO_BAUDRATE)
    print("========================================")

    arduino = serial.Serial(
        port=ARDUINO_PORT,
        baudrate=ARDUINO_BAUDRATE,
        timeout=1
    )

    time.sleep(2)

    print("Arduino Serial 연결 성공")


def send_arduino_command(command):
    global arduino

    if arduino is None:
        return False

    try:
        with arduino_lock:
            arduino.write(
                command.encode()
            )

        return True

    except Exception as e:
        print(
            "Arduino command error:",
            e
        )
        return False


# =====================================================
# Flask
# =====================================================


app = Flask(__name__)

# =====================================================
# WEB CAMERA STREAM
# =====================================================

web_frame_lock = threading.Lock()
web_frame_jpeg = None


def update_web_frame(frame):
    global web_frame_jpeg

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )

    if not ok:
        return

    with web_frame_lock:
        web_frame_jpeg = encoded.tobytes()


def generate_video_stream():
    global web_frame_jpeg

    while True:
        with web_frame_lock:
            frame = web_frame_jpeg

        if frame is None:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"\r\n"
            + frame
            + b"\r\n"
        )




# =====================================================
# Web UI
# Original UI preserved
# =====================================================

HTML = r"""
<!DOCTYPE html>
<html lang="ko">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0">

<title>TwinHull Control</title>

<style>

* {
    box-sizing: border-box;
}

html,
body {
    margin: 0;
    padding: 0;

    width: 100%;
    height: 100%;

    background: #f5f7fa;

    color: #1f2937;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    overflow: hidden;
}

.container {
    width: 100%;
    max-width: 1400px;
    height: 100vh;
    margin: 0 auto;

    display: flex;
    flex-direction: column;

    background: #f5f7fa;
}

.header h1 {
    margin: 0;

    font-size: 28px;

    font-weight: 700;

    letter-spacing: 0.5px;

    color: #111827;
}

.main {
    flex: 1;

    min-height: 0;

    display: flex;

    flex-direction: column;

    gap: 15px;

    padding: 10px 15px 15px 15px;
}

.camera-box {
    width: 100%;
    min-width: 0;
    min-height: 0;

    background: #ffffff;

    border: 1px solid #d9dee7;

    border-radius: 10px;

    padding: 4px;

    box-shadow:
        0 2px 8px rgba(0, 0, 0, 0.05);

    display: flex;

    flex-direction: column;

    align-items: center;
}




.camera {
    width: 100%;
    max-width: 1380px;
    aspect-ratio: 16 / 9;

    background: #111827;

    border-radius: 6px;

    overflow: hidden;
}

.camera img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    border-radius: 6px;
}

.controls {
    flex-shrink: 0;

    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 15px;

    padding: 0 15px 10px 15px;
}

.control-box {
    background: #ffffff;

    border: 1px solid #d9dee7;

    border-radius: 10px;

    min-height: 235px;

    padding: 15px 20px;

    box-shadow:
        0 2px 8px rgba(0, 0, 0, 0.05);

    position: relative;
}

.control-title {
    font-size: 18px;

    font-weight: 700;

    color: #1769e0;

    margin-bottom: 5px;
}

.dpad {
    position: relative;

    width: 330px;

    height: 190px;

    margin: 0 auto;
}

.control-button {
    position: absolute;

    width: 105px;

    height: 70px;

    border-radius: 10px;

    border: 1px solid #d3d9e2;

    background: #ffffff;

    color: #1769e0;

    font-size: 25px;

    font-weight: 700;

    cursor: pointer;

    user-select: none;

    -webkit-user-select: none;

    box-shadow:
        0 2px 5px rgba(0, 0, 0, 0.08);

    transition:
        background 0.08s,
        transform 0.08s,
        box-shadow 0.08s;
}

.control-button:hover {
    background: #f5f8ff;
}

.control-button:active,
.control-button.active {
    background: #e8f0ff;

    transform: scale(0.96);

    box-shadow:
        0 1px 2px rgba(0, 0, 0, 0.08);
}

.button-label {
    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    height: 100%;

    line-height: 1.0;
}

.arrow {
    font-size: 28px;

    line-height: 28px;
}

.key {
    font-size: 14px;

    margin-top: 5px;

    font-weight: 600;
}

.up {
    top: 0;

    left: 112px;
}

.left {
    top: 78px;

    left: 0;
}

.center {
    top: 78px;

    left: 112px;

    background: #22a447;

    color: #ffffff;

    border-color: #22a447;

    font-size: 17px;
}

.center:hover {
    background: #1e963f;
}

.right {
    top: 78px;

    left: 224px;
}

.down {
    top: 156px;

    left: 112px;
}

.stop {
    background: #dc2626;

    color: #ffffff;

    border-color: #dc2626;

    font-size: 17px;
}

.stop:hover {
    background: #c81e1e;
}

.keyboard-hint {
    position: absolute;

    right: 20px;

    top: 58px;

    width: 115px;

    height: 105px;

    border: 1px solid #d9dee7;

    border-radius: 8px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    background: #fafbfc;

    color: #6b7280;

    font-size: 12px;

    text-align: center;
}

.keyboard-icon {
    font-size: 24px;

    margin-bottom: 7px;

    color: #374151;
}

.keyboard-keys {
    margin-top: 6px;

    color: #1769e0;

    font-size: 15px;

    font-weight: 700;

    letter-spacing: 2px;
}

.footer {
    flex-shrink: 0;

    height: 50px;

    margin: 0 15px 15px 15px;

    padding: 0 20px;

    background: #ffffff;

    border: 1px solid #d9dee7;

    border-radius: 8px;

    display: flex;

    align-items: center;

    color: #4b5563;

    font-size: 14px;

    box-shadow:
        0 2px 8px rgba(0, 0, 0, 0.04);
}

button,
.dpad,
.control-box {
    -webkit-tap-highlight-color: transparent;
}

</style>

</head>

<body>

<div class="container">

    <div class="main">

        <div class="camera-box">

            <div class="camera">
                <img
                    id="camera-stream"
                    src="/video_feed"
                    alt="Jetson Camera / YOLO"
                >
            </div>

        </div>

    </div>

    <div class="controls">

        <div class="control-box">

            <div class="control-title">
                MOVEMENT
            </div>

            <div class="dpad">

                <button
                    class="control-button up"
                    data-command="W">

                    <div class="button-label">

                        <div class="arrow">
                            ▲
                        </div>

                        <div class="key">
                            W
                        </div>

                    </div>

                </button>

                <button
                    class="control-button left"
                    data-command="A">

                    <div class="button-label">

                        <div class="arrow">
                            ◀
                        </div>

                        <div class="key">
                            A
                        </div>

                    </div>

                </button>

                <button
                    class="control-button center stop"
                    data-command="X">

                    <div class="button-label">

                        STOP

                        <div class="key">
                            X
                        </div>

                    </div>

                </button>

                <button
                    class="control-button right"
                    data-command="D">

                    <div class="button-label">

                        <div class="arrow">
                            ▶
                        </div>

                        <div class="key">
                            D
                        </div>

                    </div>

                </button>

                <button
                    class="control-button down"
                    data-command="S">

                    <div class="button-label">

                        <div class="arrow">
                            ▼
                        </div>

                        <div class="key">
                            S
                        </div>

                    </div>

                </button>

            </div>

            <div class="keyboard-hint">

                <div class="keyboard-icon">
                    ⌨
                </div>

                KEYBOARD

                <div class="keyboard-keys">
                    W A S D
                </div>

            </div>

        </div>

        <div class="control-box">

            <div class="control-title">
                PAN / TILT
            </div>

            <div class="dpad">

                <button
                    class="control-button up"
                    data-command="I">

                    <div class="button-label">

                        <div class="arrow">
                            ▲
                        </div>

                        <div class="key">
                            I
                        </div>

                    </div>

                </button>

                <button
                    class="control-button left"
                    data-command="J">

                    <div class="button-label">

                        <div class="arrow">
                            ◀
                        </div>

                        <div class="key">
                            J
                        </div>

                    </div>

                </button>

                <button
                    class="control-button center"
                    data-command="C">

                    <div class="button-label">

                        CENTER

                        <div class="key">
                            C
                        </div>

                    </div>

                </button>

                <button
                    class="control-button right"
                    data-command="L">

                    <div class="button-label">

                        <div class="arrow">
                            ▶
                        </div>

                        <div class="key">
                            L
                        </div>

                    </div>

                </button>

                <button
                    class="control-button down"
                    data-command="K">

                    <div class="button-label">

                        <div class="arrow">
                            ▼
                        </div>

                        <div class="key">
                            K
                        </div>

                    </div>

                </button>

            </div>

            <div class="keyboard-hint">

                <div class="keyboard-icon">
                    ⌨
                </div>

                KEYBOARD

                <div class="keyboard-keys">
                    I J K L
                </div>

            </div>

        </div>

    </div>

    </div>

<script>

function sendCommand(command) {

    fetch("/command/" + command)
        .catch(function(error) {

            console.error(
                "Command error:",
                error
            );

        });

}

const buttons =
    document.querySelectorAll(
        "button[data-command]"
    );

buttons.forEach(function(button) {

    let interval = null;

    function startCommand(event) {

        if (event) {
            event.preventDefault();
        }

        const command =
            button.dataset.command;

        button.classList.add("active");

        if (command === "X") {

            sendCommand("X");
            return;

        }

        if (command === "C") {

            sendCommand("C");
            return;

        }

        sendCommand(command);

        interval =
            setInterval(
                function() {

                    sendCommand(command);

                },
                100
            );

    }

    function stopCommand(event) {

        if (event) {
            event.preventDefault();
        }

        if (interval !== null) {

            clearInterval(interval);

            interval = null;

        }

        button.classList.remove("active");

        const command =
            button.dataset.command;

        if (
            command === "W" ||
            command === "A" ||
            command === "S" ||
            command === "D"
        ) {

            sendCommand("X");

        }

    }

    button.addEventListener(
        "mousedown",
        startCommand
    );

    button.addEventListener(
        "mouseup",
        stopCommand
    );

    button.addEventListener(
        "mouseleave",
        stopCommand
    );

    button.addEventListener(
        "contextmenu",
        function(event) {

            event.preventDefault();

        }
    );

});

const keyCommands = {

    "w": "W",
    "a": "A",
    "s": "S",
    "d": "D",

    "i": "I",
    "j": "J",
    "k": "K",
    "l": "L"

};

let pressedKeys = {};

document.addEventListener(
    "keydown",
    function(event) {

        const key =
            event.key.toLowerCase();

        if (!(key in keyCommands)) {
            return;
        }

        event.preventDefault();

        if (pressedKeys[key]) {
            return;
        }

        const command =
            keyCommands[key];

        sendCommand(command);

        const interval =
            setInterval(
                function() {

                    sendCommand(command);

                },
                100
            );

        pressedKeys[key] = interval;

    }
);

document.addEventListener(
    "keyup",
    function(event) {

        const key =
            event.key.toLowerCase();

        if (!(key in keyCommands)) {
            return;
        }

        event.preventDefault();

        const interval =
            pressedKeys[key];

        if (interval) {

            clearInterval(interval);

            delete pressedKeys[key];

        }

        const command =
            keyCommands[key];

        if (
            command === "W" ||
            command === "A" ||
            command === "S" ||
            command === "D"
        ) {

            sendCommand("X");

        }

    }
);

window.addEventListener(
    "blur",
    function() {

        for (
            const key in pressedKeys
        ) {

            clearInterval(
                pressedKeys[key]
            );

        }

        pressedKeys = {};

        sendCommand("X");

    }
);

window.addEventListener(
    "beforeunload",
    function() {

        sendCommand("X");

    }
);

</script>

</body>
</html>
"""


# =====================================================
# Flask Routes
# =====================================================

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return app.response_class(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/command/<command>")
def command(command):
    allowed = [
        "W",
        "A",
        "S",
        "D",
        "X",
        "I",
        "J",
        "K",
        "L",
        "C"
    ]

    command = command.upper()

    if command in allowed:

        send_arduino_command(command)

        return "OK"

    return "INVALID", 400


# =====================================================
# Flask Thread
# =====================================================

def run_flask():
    print("========================================")
    print("TWINHULL CONTROL SYSTEM")
    print("========================================")
    print("Arduino Port :", ARDUINO_PORT)
    print("Baudrate     :", ARDUINO_BAUDRATE)
    print("Web Port     :", FLASK_PORT)
    print("----------------------------------------")
    print("WASD : Movement")
    print("IJKL : Pan / Tilt")
    print("C    : Center")
    print("X    : Stop")
    print("========================================")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


# =====================================================
# Main
# =====================================================


# =====================================================
# ArUco Marker Detection
# =====================================================

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()


def detect_aruco(frame):
    """
    Detect ArUco markers on the same camera frame used by YOLO.

    Returns:
        [
            {
                "id": int,
                "center": {"x": float, "y": float},
                "bbox": [x1, y1, x2, y2],
                "corners": [[x, y], ...]
            },
            ...
        ]
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray,
        ARUCO_DICT,
        parameters=ARUCO_PARAMS
    )

    markers = []

    if ids is None:
        return markers

    for i, marker_id in enumerate(ids.flatten()):

        pts = corners[i][0]

        x_min = float(np.min(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        x_max = float(np.max(pts[:, 0]))
        y_max = float(np.max(pts[:, 1]))

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        markers.append({
            "id": int(marker_id),
            "center": {
                "x": cx,
                "y": cy
            },
            "bbox": [
                int(x_min),
                int(y_min),
                int(x_max),
                int(y_max)
            ],
            "corners": [
                [float(p[0]), float(p[1])]
                for p in pts
            ]
        })

    return markers


def get_surface_context(markers):
    """
    현재 화면에서 검출된 ArUco marker들로
    어떤 면인지 판단하고 homography를 계산한다.

    반환:
        {
            "id": "foam-board-front/back",
            "homography": H
        }
        또는 None

    H는 카메라 픽셀 좌표 -> surface의 정규화 좌표
    [u,v] (0~1) 변환용이다.
    """

    detected = {}

    for marker in markers:
        try:
            marker_id = int(marker["id"])
            detected[marker_id] = marker
        except (KeyError, TypeError, ValueError):
            continue

    for surface_id, layout in SURFACE_MARKER_LAYOUT.items():

        if not all(
            marker_id in detected
            for marker_id in layout.keys()
        ):
            continue

        image_points = []
        surface_points = []

        for marker_id, (u, v) in layout.items():

            center = detected[marker_id]["center"]

            image_points.append([
                float(center["x"]),
                float(center["y"])
            ])

            surface_points.append([
                float(u),
                float(v)
            ])

        image_points = np.asarray(
            image_points,
            dtype=np.float32
        )

        surface_points = np.asarray(
            surface_points,
            dtype=np.float32
        )

        try:
            H, status = cv2.findHomography(
                image_points,
                surface_points,
                method=0
            )

            if H is None:
                continue

            print(
                "SURFACE: {} markers={} method=homography_4point".format(
                    surface_id,
                    len(image_points)
                )
            )

            return {
                "id": surface_id,
                "homography": H
            }

        except Exception as e:
            print(
                "Homography 계산 오류:",
                e
            )

    return None


def send_aruco_frame(ws_client, markers, camera_width, camera_height):
    """
    Send ArUco data through the already-existing WebSocket client's
    public queue without modifying its original class implementation.
    """

    if not ws_client.connected:
        return

    frame = {
        "type": "aruco_frame",
        "schemaVersion": 1,
        "source": "aruco",
        "sentAtUnixMs": int(time.time() * 1000),
        "camera": {
            "width": int(camera_width),
            "height": int(camera_height)
        },
        "markers": markers
    }

    message = json.dumps(
        frame,
        separators=(",", ":")
    )

    try:
        ws_client.message_queue.put_nowait(message)
    except queue.Full:
        pass


def main():

    print("========================================")
    print("      TWINHULL JETSON NANO")
    print("========================================")
    print("WebSocket 대상:", WEBSOCKET_URL)
    print("Flask         : http://0.0.0.0:{}".format(
        FLASK_PORT
    ))
    print("========================================")

    # -------------------------------------------------
    # Arduino
    # -------------------------------------------------

    init_arduino()

    # -------------------------------------------------
    # Flask
    # -------------------------------------------------

    flask_thread = threading.Thread(
        target=run_flask
    )
    flask_thread.daemon = True
    flask_thread.start()

    print("Flask 서버 thread 시작")

    # -------------------------------------------------
    # WebSocket
    # -------------------------------------------------

    ws_client = WebSocketClient(
        WEBSOCKET_URL
    )

    # -------------------------------------------------
    # TensorRT
    # -------------------------------------------------

    print(
        "Loading TensorRT engine:",
        ENGINE_PATH
    )

    detector = TensorRTEngine(
        ENGINE_PATH
    )

    # -------------------------------------------------
    # Camera
    # -------------------------------------------------

    pipeline = gstreamer_pipeline(
        capture_width=CAMERA_WIDTH,
        capture_height=CAMERA_HEIGHT,
        display_width=DISPLAY_WIDTH,
        display_height=DISPLAY_HEIGHT,
        framerate=30,
        flip_method=0,
    )

    print("GStreamer pipeline:")
    print(pipeline)

    cap = cv2.VideoCapture(
        pipeline,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():

        ws_client.close()

        if arduino is not None:
            try:
                arduino.close()
            except Exception:
                pass

        raise RuntimeError(
            "IMX219 카메라를 열지 못했습니다. "
            "GStreamer/CSI 연결을 확인하세요."
        )

    prev_time = time.time()
    fps = 0.0

    print(
        "카메라 시작. 'q'를 누르면 종료합니다."
    )

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "카메라 프레임을 읽지 못했습니다."
                )

                break

            # -------------------------------------------------
            # YOLO inference
            # -------------------------------------------------

            output = detector.infer(
                frame
            )

            detections = postprocess(
                output,
                frame.shape[1],
                frame.shape[0]
            )

            # -------------------------------------------------
            # ArUco detection
            # YOLO와 동일한 원본 카메라 프레임에서 동시에 검출
            # -------------------------------------------------

            aruco_markers = detect_aruco(frame)

            if aruco_markers:
                print(
                    "ArUco:",
                    [
                        {
                            "id": marker["id"],
                            "center": marker["center"]
                        }
                        for marker in aruco_markers
                    ]
                )

            # ArUco 데이터는 기존 WebSocketClient 클래스는 건드리지 않고
            # 별도의 aruco_frame 메시지로 같은 WebSocket 연결을 사용한다.
            send_aruco_frame(
                ws_client,
                aruco_markers,
                frame.shape[1],
                frame.shape[0]
            )

            # -------------------------------------------------
            # Jetson -> Laptop WebSocket
            # -------------------------------------------------

            # -------------------------------------------------
            # ArUco 4점 -> 현재 촬영 면 + Homography
            # -------------------------------------------------
            surface_context = get_surface_context(
                aruco_markers
            )

            # -------------------------------------------------
            # Jetson -> Laptop WebSocket
            # YOLO crack bbox + surface.id/u/v
            # -------------------------------------------------
            ws_client.send_detection_frame(
                detections,
                frame.shape[1],
                frame.shape[0],
                surface_context
            )

            # -------------------------------------------------
            # Local visualization
            # -------------------------------------------------

            for (
                x1,
                y1,
                x2,
                y2,
                conf
            ) in detections:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    "CRACK {:.2f}".format(
                        conf
                    )
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(20, y1 - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # -------------------------------------------------
            # ArUco visualization for Flask web stream
            # -------------------------------------------------

            for marker in aruco_markers:

                pts = np.array(
                    marker["corners"],
                    dtype=np.int32
                )

                cv2.polylines(
                    frame,
                    [pts],
                    True,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "ArUco ID {}".format(marker["id"]),
                    (
                        int(marker["center"]["x"]),
                        max(
                            20,
                            int(marker["center"]["y"])
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # -------------------------------------------------
            # FPS
            # -------------------------------------------------

            now = time.time()
            dt = now - prev_time
            prev_time = now

            if dt > 0:

                current_fps = 1.0 / dt

                fps = (
                    0.9 * fps +
                    0.1 * current_fps
                    if fps > 0
                    else current_fps
                )

            cv2.putText(
                frame,
                "FPS: {:.1f}  Detections: {}".format(
                    fps,
                    len(detections)
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Jetson 모니터와 동일한 최종 처리 프레임을 웹으로 전송
            update_web_frame(frame)




    except KeyboardInterrupt:

        print(
            "\n사용자에 의해 종료됩니다."
        )

    finally:

        # -------------------------------------------------
        # Emergency stop
        # -------------------------------------------------

        try:
            send_arduino_command("X")
        except Exception:
            pass

        # -------------------------------------------------
        # Camera cleanup
        # -------------------------------------------------

        cap.release()


        # -------------------------------------------------
        # WebSocket cleanup
        # -------------------------------------------------

        ws_client.close()

        # -------------------------------------------------
        # Arduino cleanup
        # -------------------------------------------------

        if arduino is not None:

            try:
                arduino.close()
            except Exception:
                pass

        print(
            "TwinHull Jetson Nano 종료 완료"
        )


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    main()
