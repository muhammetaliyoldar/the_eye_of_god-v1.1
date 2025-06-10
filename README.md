# The Eye of God - AI-Powered Traffic Violation Detection System

An advanced, real-time traffic analysis and violation detection system built on state-of-the-art computer vision models.

> **Note:** This project is currently under active development. Features are being continuously improved, and the system is evolving. The documentation reflects the current state but is subject to change.

## Core Features

- **Multi-Violation Detection**: Identifies both traffic light violations (running red/yellow lights) and speeding infractions.
- **High-Precision Recognition**: Utilizes a dual-YOLOv8 model strategy for robust vehicle/light detection and specialized, high-accuracy license plate reading.
- **Custom-Trained Models**: The AI models were trained on a custom dataset collected by drone, achieving over 97.3% mAP for superior performance in real-world scenarios.
- **Real-time Speed Estimation**: Calculates the speed of tracked vehicles with high accuracy using perspective transformation from a calibrated matrix.
- **Automated Evidence Archiving**: Automatically saves detailed image evidence of violations, including vehicle type, speed, and license plate.
- **SQLite Database Integration**: Logs all detected vehicles and violations for data analysis and record-keeping.

## Showcase

Here are some sample outputs from the project, demonstrating the system's capabilities in different scenarios.

### Vehicle Detection, Speed Estimation & Classification
These images show the system detecting vehicles, tracking them with unique IDs, estimating their real-time speed, and classifying their type (e.g., Car, Truck).

<table align="center">
<tr>
    <td><img src="show/screenshots/speed-car_calissificiation/speed-car_calissificiation (1).png" alt="Speed and Classification 1" width="100%"></td>
    <td><img src="show/screenshots/speed-car_calissificiation/speed-car_calissificiation (2).png" alt="Speed and Classification 2" width="100%"></td>
</tr>
<tr>
    <td><img src="show/screenshots/speed-car_calissificiation/speed-car_calissificiation (3).png" alt="Speed and Classification 3" width="100%"></td>
    <td><img src="show/screenshots/speed-car_calissificiation/speed-car_calissificiation (4).png" alt="Speed and Classification 4" width="100%"></td>
</tr>
</table>

### License Plate & Traffic Light Detection
This section demonstrates the detection of license plates and the status of traffic lights, which are crucial for identifying violations.

<table align="center">
<tr>
    <td><img src="show/screenshots/lisence_plate-traffic_lights/lisence_plate-traffic_lights (1).png" alt="License Plate and Traffic Lights 1" width="100%"></td>
    <td><img src="show/screenshots/lisence_plate-traffic_lights/lisence_plate-traffic_lights (2).png" alt="License Plate and Traffic Lights 2" width="100%"></td>
</tr>
</table>

### Violation Evidence Output
When a violation is detected, the system saves detailed evidence. These images are examples of the final output files, showing the offending vehicle, its license plate, speed, and the violation type.

<table align="center">
<tr>
    <td><img src="show/screenshots/outputs/outputs (2).png" alt="Output Evidence 1" width="100%"></td>
    <td><img src="show/screenshots/outputs/outputs (1).png" alt="Output Evidence 2" width="100%"></td>
</tr>
<tr>
    <td><img src="show/screenshots/outputs/outputs (4).png" alt="Output Evidence 3" width="100%"></td>
    <td><img src="show/screenshots/outputs/outputs (3).png" alt="Output Evidence 4" width="100%"></td>
</tr>
</table>

## Model Training & Strategy

The high performance of this system is a direct result of a meticulous, custom training process.

1.  **Data Collection**: A unique dataset was created by recording high-resolution video footage of traffic intersections using a drone.
2.  **Frame Extraction & Annotation**: The video was decomposed into thousands of individual frames. These frames were then painstakingly annotated with the following classes: `car`, `truck`, `bus`, `motorcycle`, `green traffic light`, `red traffic light`, `yellow traffic light`, and `license plate`.
3.  **Model Selection & Training**: Several architectures were evaluated, including YOLOv12. However, **YOLOv8** was selected for its superior balance of speed and accuracy on this specific task. The model was trained on the custom dataset in a local environment, achieving the following benchmark scores:
    *   **mAP50-95:** 0.973
    *   **F1-Score:** 0.96
4.  **Dual-Model Strategy**: To maximize accuracy, two separate YOLOv8 models are used:
    *   **General Detection Model**: A robust model trained on all classes for general scene understanding (vehicle tracking, traffic light status).
    *   **Specialized Plate Model**: A second model fine-tuned exclusively on license plate data. This ensures extremely high precision in license plate recognition, which is critical for evidence generation.

## Project Structure

The project is organized as follows:

```
the_eye_of_god-v1.1/
├── database/              # SQLite database files
├── evidence/              # Stores violation evidence sub-folders
├── models/                # Pre-trained YOLOv8 models (not in repo)
├── outputs/               # General output files (e.g., processed videos)
├── show/                  # Assets for documentation (screenshots)
├── utils/                 # Helper scripts and utility functions
├── video-data/            # Raw video files (not in repo)
├── .gitignore             # Specifies intentionally untracked files
├── main.py                # Main application script
├── PRD.md                 # Product Requirements Document
├── README.md              # This file
└── requirements.txt       # Python package requirements
```

## System & Calibration

### System Requirements
- Python 3.8-3.10
- PyTorch & CUDA-enabled GPU (recommended for performance)
- Tesseract OCR

### Calibration Matrix
The system uses a perspective transformation matrix for accurate speed calculation. The source points for this matrix are:
`[(912, 215), (1321, 209), (1918, 608), (424, 610), (653, 430), (1642, 427)]`

## Installation

```bash
# Clone the repository
git clone https://github.com/muhammetaliyoldar/the_eye_of_god-v1.1.git
cd the_eye_of_god-v1.1

# Create and activate a virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Access to Data & Models

The custom-collected video dataset and the final trained model weights (`.pt` files) are not included in this repository due to their large size. If you are interested in accessing these assets for research, collaboration, or evaluation purposes, please reach out.

- **Email**: `muhammetaliyoldar@gmail.com`
- **LinkedIn**: `https://www.linkedin.com/in/muhammet-ali-yoldar/`

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 