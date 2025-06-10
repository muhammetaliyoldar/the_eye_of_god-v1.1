# Product Requirements Document (PRD) - The Eye of God

**Author**: [Your Name/Team Name]
**Status**: In Development
**Last Updated**: [Current Date]

---

## 1. Introduction & Vision

"The Eye of God" is an AI-powered traffic analysis system designed to autonomously detect and document traffic violations. By leveraging custom-trained deep learning models, the system aims to provide a reliable, efficient, and scalable solution for improving traffic safety and enforcing regulations. This document outlines the product requirements for the system, which is currently in an active development phase.

## 2. Core Objectives & Goals

- **Primary Goal**: To accurately detect traffic light and speeding violations in real-time from video feeds.
- **Secondary Goal**: To create a robust, verifiable evidence package for each detected violation.
- **Business Goal**: To provide a foundational technology for smart city infrastructure, targeting municipalities and traffic enforcement agencies.
- **Technical Goal**: To continuously improve model accuracy and expand the system's capabilities to include other violation types.

## 3. Target Audience

- **Primary Users**: Traffic Enforcement Agencies, Municipal Police Departments.
- **Secondary Users**: Smart City System Integrators, Urban Planning Departments, Transportation Researchers.

## 4. Features & Scope

### 4.1. Core Functionality (Current Implementation)
- **Real-time Video Processing**: Ingests and analyzes 4K video streams.
- **Vehicle Detection & Tracking**: Identifies cars, trucks, buses, and motorcycles, assigning a persistent ID to each.
- **Traffic Light State Recognition**: Detects traffic lights and accurately determines their state (red, yellow, green).
- **License Plate Recognition**: A specialized model detects and reads license plates with high precision.
- **Speed Estimation**: Calculates vehicle speed using a calibrated perspective transformation.
- **Violation Logic**: Flags violations based on pre-defined rules (e.g., crossing a line on red, exceeding the speed limit).
- **Evidence Generation**: Saves a snapshot image and a text file with violation details (plate, speed, timestamp, etc.).
- **Database Logging**: Records all events in an SQLite database.

### 4.2. Future Development (Roadmap)
- Multi-camera support for comprehensive intersection coverage.
- A web-based dashboard for reviewing violations and analytics.
- Expansion to other violation types (e.g., illegal lane changes, parking violations).
- Cloud-based deployment for enhanced scalability.

## 5. Technical Architecture & Stack

### 5.1. AI & Computer Vision
- **Core Architecture**: Dual YOLOv8 models.
  - **Model 1 (General):** Trained on a comprehensive custom dataset for detecting vehicles and traffic lights.
  - **Model 2 (Specialized):** Fine-tuned specifically for high-accuracy license plate detection.
- **OCR Engine**: Tesseract OCR for converting license plate images to text.
- **Core Libraries**: OpenCV, PyTorch, NumPy, Supervision.

### 5.2. Data Management
- **Database**: SQLite for local, structured data logging.
- **File System**: A structured `evidence/` directory to store image and text files for each violation, organized by license plate.

### 5.3. System & Environment
- **Language**: Python 3.10
- **Environment**: Virtual environment with dependencies managed via `requirements.txt`.
- **Recommended Hardware**: CUDA-enabled NVIDIA GPU (for real-time performance).

## 6. System Requirements & Performance

- **Input**: 4K video stream @ 30 FPS.
- **Processing Target**: Minimum 25 FPS on recommended hardware.
- **Model Accuracy**:
  - **mAP50-95 (Overall):** > 97%
  - **License Plate Recognition Accuracy:** > 95% (end-to-end OCR).
- **Data Integrity**: All generated evidence must be timestamped and securely stored.

## 7. Data & Models Access

The proprietary dataset (drone footage) and trained model weights are not publicly available in the repository. Access can be granted for academic or collaborative purposes by contacting the project owner.

- **Contact Email**: `muhammetaliyoldar@gmail.com`
- **Contact LinkedIn**: `https://www.linkedin.com/in/muhammet-ali-yoldar/`

## 8. Assumptions & Dependencies

- The system requires a fixed-camera perspective for the calibration matrix to be valid.
- Performance is highly dependent on the available hardware, particularly the GPU.
- Tesseract OCR must be installed on the host system.

## 9. Violation Detection Criteria

### 9.1 Traffic Light Violation
- Vehicles crossing the virtual line when the light is not green (red, yellow, or red+yellow).
- Virtual line: Between coordinates [856, 247] and [1179, 245].
- Light color to be verified by HSV analysis.

### 9.2 Speeding Violation
- Urban speed limits:
  - Car: 50 km/h
  - Motorcycle: 50 km/h
  - Van: 50 km/h
  - Truck: 40 km/h
  - Bus: 50 km/h
- Perspective transformation will be applied using matrix coordinates.
- Speed will be calculated for vehicles tracked for at least 3 seconds.

## 10. Output Formats

### 10.1 Violation Record File Structure
```
violation_records/
└── [LICENSE_PLATE]/
    ├── violation_[ID]_[DATE]_[TIME].jpg  # Violation snapshot
    └── violation_[ID]_[DATE]_[TIME].txt  # Violation details
```

### 10.2 Database Schema
- **vehicles**: id, plate, city, vehicle_type, first_seen, last_seen
- **violations**: id, vehicle_id, violation_type, speed, speed_limit, light_color, timestamp, evidence_path

## 11. Implementation Process

1. System calibration with a test video.
2. Integration with real traffic cameras.
3. 3-month pilot implementation.
4. Transition to full-scale deployment.

## 12. Expansion Plans

- Multi-camera support.
- Cloud-based central system.
- Web interface and mobile application.
- Continuous improvement of AI models.
- Detection of additional violation types (emergency lane violation, parking violation, etc.).

## 13. Additional Information

This document has been prepared to detail the scope and requirements of "The Eye of God" project. The project was developed to increase traffic safety and detect violations more effectively. For more information, please contact the project manager. 