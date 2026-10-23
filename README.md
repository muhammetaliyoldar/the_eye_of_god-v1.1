# The Eye of God: Traffic Violation Detection

A Python and computer‑vision prototype that analyzes fixed‑angle traffic video to detect **red/yellow‑light** and **speeding** violations. It detects vehicles with YOLOv8 (Ultralytics), tracks them with ByteTrack (Supervision), estimates speed through a perspective (homography) transform, reads Turkish license plates with Tesseract OCR, and logs vehicles and violations to SQLite together with image and text evidence.

> **Status:** prototype or work in progress. Model weights and the source video are not in the repository, and the committed `main.py` has syntax errors (see [Project Status](#project-status)).

## Features

- **Vehicle detection and tracking**: a YOLOv8 model (`models/yolov8x.pt`) with `sv.ByteTrack`. Vehicles are filtered to `car`, `truck`, `bus` and `motorcycle` using COCO class IDs.
- **Traffic lights and license plates**: a second YOLOv8 model (`models/yolov8y.pt`) detects traffic lights and plates (class IDs 9 and 1 in `main.py`). An HSV‑based `TrafficLightColorDetector` (`utils/color_detection.py`) determines the light colour, and red+yellow is treated as red.
- **Speed estimation**: `utils/speed_estimation.py` maps tracked points to a bird's‑eye plane via `cv2.getPerspectiveTransform` and converts displacement over time to km/h. Default limits: 50 km/h (car, bus, motorcycle) and 40 km/h (truck).
- **Violation logic**: `utils/violation_check.py` covers red‑light crossing of a stop line (`CROSS_LINE`), yellow‑light crossing and speeding, and suppresses duplicate reports per vehicle.
- **Plate OCR**: `utils/ocr.py` runs preprocessing (grayscale, blur, adaptive threshold) followed by Tesseract, cleans the text into the Turkish plate format and maps the province code (01–81) to a city name.
- **Evidence and logging**: saves a JPG and TXT per violation under `outputs/ihlal_kayitlari/<plate>/` and writes `vehicles` / `violations` tables to `database/violations.db` (SQLite). Application logs go to `outputs/logs/application.log`.
- **Mock fallback**: if the YOLO weights cannot be loaded, `utils/mock_models.py` supplies random detections so that the pipeline can still be exercised.

## Screenshots

Annotated frames from the drone/elevated test footage (the files are in `show/screenshots/`).

**Speed estimation and vehicle classification**

<table>
<tr>
  <td><img src="show/screenshots/speed-car_calissificiation/speed-car_calissificiation%20(1).png" width="100%"></td>
  <td><img src="show/screenshots/speed-car_calissificiation/speed-car_calissificiation%20(2).png" width="100%"></td>
</tr>
</table>

**License plate and traffic‑light detection**

<table>
<tr>
  <td><img src="show/screenshots/lisence_plate-traffic_lights/lisence_plate-traffic_lights%20(1).png" width="100%"></td>
  <td><img src="show/screenshots/lisence_plate-traffic_lights/lisence_plate-traffic_lights%20(2).png" width="100%"></td>
</tr>
</table>

**Combined output**

<table>
<tr>
  <td><img src="show/screenshots/final/final%20(1).png" width="100%"></td>
  <td><img src="show/screenshots/final/final%20(2).png" width="100%"></td>
</tr>
</table>

## Data and Models

- **Video**: the scripts expect `video-data/test-video.mp4`. `*.mp4` is git‑ignored, so the video is **not** in the repository. `video-data/kordinat.png` is a reference frame used for calibration.
- **Weights**: `models/yolov8x.pt` (vehicles) and `models/yolov8y.pt` (traffic lights and plates). `*.pt` is git‑ignored, so the weights are **not** in the repository.
- **Training**: according to the previous version of this README, the models were trained on frames extracted from drone footage of intersections, annotated with `car`, `truck`, `bus`, `motorcycle`, `green/red/yellow traffic light` and `license plate`. It also reported mAP50‑95 = 0.973 and F1 = 0.96. Neither the training code nor the logs, metrics or dataset are in this repository, so those figures cannot be verified here.

The author offers access to the video and weights for research or collaboration on request: muhammetaliyoldar@gmail.com · [LinkedIn](https://www.linkedin.com/in/muhammet-ali-yoldar/).

## Calibration

| Item | Value | Where |
|------|-------|-------|
| Perspective source polygon (4 points) | `[(941,228), (1284,219), (1905,647), (451,718)]` | `main.py`, `app-2.py`, `show/*.py` |
| Real‑world target size | 20 × 30 (`main.py`), 11 × 25 (`app-2.py`, show scripts) | `TARGET_WIDTH/HEIGHT` |
| Stop line | `[[856,247], [1179,245]]` | `main.py` `CROSS_LINE` |
| 6‑point picker output | `[(912,215), (1321,209), (1918,608), (424,610), (653,430), (1642,427)]` | `koordinatlar.txt` (from `show/coordinate_picker.py`) |

The points are specific to the test video. For a new camera, select the points again with `show/coordinate_picker.py`.

## Tech Stack

Python · Ultralytics YOLOv8 (`ultralytics==8.0.225`) · Supervision 0.16 (ByteTrack, annotators, `PolygonZone`) · OpenCV · PyTorch 2.0.1 · pytesseract / Tesseract OCR · NumPy · pandas · SQLite.

`requirements.txt` also lists `deep_sort_realtime`, `imutils` and `scikit-image`, but none of the committed code imports them.

## Project Structure

```
the_eye_of_god-v1.1/
├── main.py                 # full pipeline: detection, tracking, speed, light, OCR, violations, DB
├── app-2.py                # standalone plate + traffic-light + speed demo (single model, yolov8y.pt)
├── show/
│   ├── speed-show.py       # speed + vehicle class demo (yolov8x.pt)
│   ├── lisenceplate-trafficlight-show.py
│   ├── coordinate_picker.py  # click 6 calibration points on a frame
│   └── screenshots/        # images used in this README
├── utils/
│   ├── speed_estimation.py   homography_utils.py   color_detection.py
│   ├── violation_check.py    database.py           ocr.py
│   └── mock_models.py
├── video-data/kordinat.png # calibration reference frame
├── koordinatlar.txt        # picked calibration points
├── PRD.md                  # product requirements document
├── requirements.txt
└── LICENSE                 # MIT
```

Generated at runtime and git‑ignored: `outputs/`, `database/`, `*.db`, `*.log`.

## Installation

```bash
git clone https://github.com/muhammetaliyoldar/the_eye_of_god-v1.1.git
cd the_eye_of_god-v1.1
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You also need the following:

- **Tesseract OCR**. `utils/ocr.py` hard‑codes `C:\Program Files\Tesseract-OCR\tesseract.exe`, so change it on other systems.
- A CUDA GPU is recommended, but the code falls back to the CPU.
- Place the weights in `models/` and the video in `video-data/`.

## Usage

All scripts use **absolute Windows paths** (`C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\...`) for the video, the output and the weights. Edit `SOURCE_VIDEO_PATH`, `TARGET_VIDEO_PATH` and the model paths at the top of each script first.

```bash
python main.py                                  # full pipeline (fix the syntax errors first, see below)
python app-2.py                                 # plate + traffic light + speed demo, press q to quit
python show/speed-show.py                       # speed / classification demo
python show/lisenceplate-trafficlight-show.py   # plate / traffic-light demo
python show/coordinate_picker.py                # pick calibration points
```

The output videos are written to `outputs/` or `show/*-output/`, and the evidence is saved under `outputs/ihlal_kayitlari/`.

## Project Status

- Development took place between **6 and 10 June 2025** (11 commits), and there has been no further activity since.
- The committed `main.py` does not compile. Line 1 reads `qimport cv2` instead of `import cv2`, and the `try:` block around `sv.VideoInfo.from_video_path` (around line 153) is mis‑indented. `show/speed-show.py` also has a syntax error near line 92. `app-2.py` and `show/lisenceplate-trafficlight-show.py` compile.
- `app.py` is intentionally git‑ignored ("application files that should not be public").
- `PRD.md` describes the target product and its roadmap: multi‑camera support, a web dashboard, more violation types and cloud deployment. These items are **not** implemented.

## License

MIT, see [LICENSE](LICENSE).
