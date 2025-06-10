# Product Requirements Document (PRD) - The Eye of God

## 1. Introduction

"The Eye of God" is an AI-powered traffic violation detection system. The system analyzes images from traffic cameras to detect, record, and archive traffic light violations and speed limit infringements.

## 2. Product Purpose

To enhance traffic safety and promote compliance with traffic rules by:
- Detecting traffic light violations (running red or yellow lights).
- Measuring vehicle speeds to identify speeding violations.
- Archiving detected violations with their evidence.
- Making traffic enforcement more effective and fair.

## 3. Target Users

- Traffic Enforcement Units
- Municipal Traffic Management Centers
- General Directorate of Highways
- Smart City Management Systems

## 4. System Components

### 4.1 Input Sources

- **Video Source**: 4K resolution, 30 FPS traffic camera footage.
- **AI Models**:
  - YOLOv8x: For vehicle detection and classification.
  - YOLOv8y: For traffic light and license plate region detection.
- **OCR Engine**: Tesseract OCR (with Turkish language support).

### 4.2 Core Functions

#### Detection and Recognition
- Vehicle detection and classification (car, truck, motorcycle, etc.).
- Traffic light detection and color analysis.
- License plate region detection and OCR for plate reading.
- Vehicle tracking and assignment of unique IDs.

#### Violation Analysis
- Traffic light violation detection (crossing on red/yellow light).
- Vehicle speed calculation using matrix transformation.
- Speed limit check based on vehicle type.
- Evidence generation in case of a violation.

#### Data Management
- Structured filing system for violation records.
- Storage of vehicle and violation information in an SQLite database.
- City mapping according to the Turkish license plate system.

#### Visualization
- Real-time detection and tracking visualization.
- On-screen display of vehicle information and speed values.
- Color-coded display of traffic light status.

## 5. Technical Requirements

### 5.1 Hardware
- NVIDIA GPU with CUDA support (Minimum 6GB VRAM).
- 16GB+ RAM.
- 500GB+ storage space (for video archive).

### 5.2 Software
- Python 3.10
- YOLOv8 models
- Tesseract OCR (with Turkish language pack).
- CUDA and cuDNN (for GPU usage).
- OpenCV, NumPy, PyTorch.

### 5.3 Performance Requirements
- Real-time processing of 4K video stream (minimum 25 FPS).
- 95%+ accuracy in vehicle detection.
- 90%+ accuracy in license plate reading.
- 98%+ accuracy in traffic light color analysis.

## 6. Violation Detection Criteria

### 6.1 Traffic Light Violation
- Vehicles crossing the virtual line when the light is not green (red, yellow, or red+yellow).
- Virtual line: Between coordinates [856, 247] and [1179, 245].
- Light color to be verified by HSV analysis.

### 6.2 Speeding Violation
- Urban speed limits:
  - Car: 50 km/h
  - Motorcycle: 50 km/h
  - Van: 50 km/h
  - Truck: 40 km/h
  - Bus: 50 km/h
- Perspective transformation will be applied using matrix coordinates.
- Speed will be calculated for vehicles tracked for at least 3 seconds.

## 7. Output Formats

### 7.1 Violation Record File Structure
```
violation_records/
└── [LICENSE_PLATE]/
    ├── violation_[ID]_[DATE]_[TIME].jpg  # Violation snapshot
    └── violation_[ID]_[DATE]_[TIME].txt  # Violation details
```

### 7.2 Database Schema
- **vehicles**: id, plate, city, vehicle_type, first_seen, last_seen
- **violations**: id, vehicle_id, violation_type, speed, speed_limit, light_color, timestamp, evidence_path

## 8. Implementation Process

1. System calibration with a test video.
2. Integration with real traffic cameras.
3. 3-month pilot implementation.
4. Transition to full-scale deployment.

## 9. Expansion Plans

- Multi-camera support.
- Cloud-based central system.
- Web interface and mobile application.
- Continuous improvement of AI models.
- Detection of additional violation types (emergency lane violation, parking violation, etc.).

## 10. Additional Information

This document has been prepared to detail the scope and requirements of "The Eye of God" project. The project was developed to increase traffic safety and detect violations more effectively. For more information, please contact the project manager. 