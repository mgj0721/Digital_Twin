# Digital_Twin

이동형 검사 모듈로 선체 표면을 촬영하면, Jetson Nano가 YOLOv5로 부식을 찾아내고
ArUco 마커로 위치를 계산해서 Unity 3D 디지털 트윈 위에 실시간으로 표시해준다.
하기는 시스템 아케텍처와 각 폴더별 역할, 데이터의 흐름과 실행 순서를 상술했다.
## 1. 전체 아키텍처

```mermaid
flowchart LR
    subgraph Field["실제 선체 표면"]
        Cam["카메라\n(이동형 검사 모듈)"]
    end

    subgraph JetsonBox["Jetson Nano — Jetson_package/"]
        YOLO["YOLOv5 (TensorRT)\n부식 탐지"]
        ArUco["ArUco 인식 +\nhomography 계산\n(u, v 좌표 변환)"]
        WSClient["WebSocket 클라이언트\n(\"JETSON\"으로 식별)"]
        Flask["Flask 웹 컨트롤러\n(모듈 원격 조종 UI)"]
    end

    subgraph ArduinoBox["Arduino — Arduino/"]
        Motor["DC 모터 + 서보\n(이동 / 팬틸트)"]
    end

    subgraph LaptopBox["노트북 — Laptop_server/"]
        Relay["WebSocket 서버\n(중계 전용)"]
    end

    subgraph UnityBox["Unity PC — Unity_Script/"]
        Receiver["JetsonWebSocketReceiver.cs\nWS 클라이언트 + JSON 파싱"]
        Visualizer["DigitalTwinDetectionVisualizer.cs\n2D bbox 미리보기 (conf ≥ 0.50)"]
        Mapper["SurfaceMarkerMapper.cs\n3D Marker 배치 (conf ≥ 0.70)"]
    end

    Cam --> YOLO
    Cam --> ArUco
    YOLO --> WSClient
    ArUco --> WSClient
    Flask <--> Motor
    WSClient -- "detection_frame JSON\n(WebSocket)" --> Relay
    Relay -- "JETSON → UNITY 중계" --> Receiver
    Receiver --> Visualizer
    Receiver --> Mapper
```

`yolo/` 폴더는 이 실시간 파이프라인에는 안 들어간다. Jetson에 올라가는 모델
(`best.pt`/엔진 파일)을 만들어낸 **학습 과정**(Colab)을 따로 담아둔 폴더다.

## 2. 폴더별 역할

| 폴더 | 실행 위치 | 역할 |
| --- | --- | --- |
| **`Arduino/`** | 이동형 검사 모듈에 장착된 Arduino | Jetson의 Flask 컨트롤러가 시리얼로 보내는 명령을 받아 실제로 모듈을 움직인다. L298N 드라이버로 DC 모터 2개를 제어해서 전후좌우 이동(W/A/S/D)을, 서보모터 2개로 카메라 팬/틸트(I/J/K/L, C는 중앙 정렬)를 맡는다. 명령이 300ms 넘게 안 들어오면 자동으로 멈추는 안전장치도 들어있다. |
| **`Jetson_package/`** | Jetson Nano | 카메라 프레임을 받아서 ① TensorRT로 최적화한 YOLOv5로 부식을 탐지하고, ② ArUco 마커를 인식해서 homography를 계산해 탐지 위치를 표면 좌표(u, v)로 바꾸고, ③ 그 결과를 표준 JSON으로 묶어 WebSocket으로 노트북에 보낸다. 모듈을 원격 조종하는 Flask 웹 UI도 여기서 같이 돌아간다. |
| **`Laptop_server/`** | 노트북 (Jetson·Unity와 같은 Wi-Fi) | 딱 WebSocket **중계**만 하는 서버. Jetson과 Unity가 각각 접속해서 첫 메시지로 자기가 누군지 밝히면(“JETSON”/“UNITY”), Jetson이 보낸 탐지 결과를 그대로 Unity로 넘겨준다. |
| **`Unity_Script/`** | Unity PC | 중계받은 JSON을 디지털 트윈에 실제로 반영하는 부분. `JetsonWebSocketReceiver.cs`가 수신과 역직렬화를 맡고, `DigitalTwinDetectionVisualizer.cs`는 화면에 2D 박스로 미리 보여주고, `SurfaceMarkerMapper.cs`는 표면 좌표(u, v)를 3D 모델 위 실제 위치로 바꿔서 부식 Marker를 배치한다. |
| **`yolo/`** | 학습용 (Colab, Jetson과는 별개) | 실제로 쓰이는 YOLOv5 모델을 학습시킨 원본 코드와 데이터 정의. 지금 학습된 모델은 **corrosion(부식) 단일 클래스**만 탐지한다. |

## 3. 데이터가 흘러가는 경로

```
카메라 프레임
  → YOLOv5(TensorRT) 추론 → bbox + confidence + label
  → ArUco 마커 인식 → homography 계산 → bbox 중심점을 표면 좌표(u, v)로 변환
  → 표준 JSON(detection_frame)으로 조립
  → WebSocket 전송: Jetson → 노트북(중계) → Unity
  → Unity에서 confidence 기준으로 2D 미리보기 / 3D 확정 배치로 나뉨
```

Jetson과 Unity가 주고받는 메시지는 이런 구조를 쓴다 (`detection_frame`):

```json
{
  "type": "detection_frame",
  "schemaVersion": 1,
  "source": "yolov5",
  "sequence": 8015,
  "sentAtUnixMs": 1780000000000,
  "camera": { "width": 1280, "height": 720 },
  "detections": [
    {
      "id": "det-0",
      "label": "corrosion",
      "confidence": 0.78,
      "bbox": { "x": 191, "y": 326, "width": 250, "height": 69 },
      "surface": { "id": "foam-board-front", "u": 0.42, "v": 0.68 }
    }
  ]
}
```

- `bbox`는 이미지 픽셀 좌표고, `surface.u`/`v`는 ArUco 마커로 계산한 **표면 위 상대 좌표(0~1)**다.
- 그 프레임에서 ArUco 마커가 인식되지 않으면 `surface` 필드는 아예 안 붙는다 — 이때도 2D 박스는 계속 뜨고, 3D Marker만 그 프레임에서 안 찍힌다.
- `confidence`는 두 단계로 쓴다: 2D 화면 미리보기는 `≥ 0.50`, 3D 디지털 트윈 확정 배치는 `≥ 0.70`.

## 4. 실행 순서

1. **노트북**: `Laptop_server/`의 WebSocket 서버부터 켜둔다 (Jetson·Unity가 붙을 준비 상태로).
2. **Jetson Nano**: `Jetson_package/`의 메인 스크립트를 실행하면 카메라·YOLOv5·ArUco·Arduino 제어가 한 번에 시작된다. 실행 전 노트북의 Wi-Fi IP를 코드 안 접속 주소에 맞게 바꿔줘야 한다.
3. **Unity**: `Unity_Script/`가 붙은 씬을 Play. `JetsonWebSocketReceiver`의 Laptop Ip/Port를 노트북 IP에 맞추고, `SurfaceMarkerMapper`의 Surface Planes에 검사 대상 표면(앞면/뒷면 등)의 기준점을 미리 등록해둬야 3D Marker가 실제로 배치된다.
4. **Arduino**: `Arduino/`의 스케치를 업로드해서 이동형 검사 모듈에 연결한다. 이후엔 Jetson의 Flask 컨트롤러(웹 브라우저)로 원격 조종한다.
