from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time

# --- VIDEO LOCATIONS ---
SOURCE_VIDEO_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\video-data\test-video.mp4"
TARGET_VIDEO_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\show\app-output\app-output.mp4"

# --- CALIBRATION POINTS (ADJUST ACCORDING TO YOUR VIDEO!) ---
SOURCE = np.array([(941, 228), (1284, 219), (1905, 647), (451, 718)])
TARGET_WIDTH = 11
TARGET_HEIGHT = 25
TARGET = np.array(
    [[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]]
)

# --- CLASS IDS ---
# Class ID mapping based on YOLOv8y
LICENSE_PLATE_CLASS_ID = 1  # License plate class ID 
PERSON_CLASS_ID = 0  # Person class ID
TRAFFIC_LIGHT_CLASS_ID = 9  # Common traffic light class ID
GREEN_LIGHT_CLASS_ID = 11   # Green light class ID - based on detection output
RED_LIGHT_CLASS_ID = 12     # Red light class ID - assumed
YELLOW_LIGHT_CLASS_ID = 13  # Yellow light class ID - assumed

# Traffic light related class IDs to check
TRAFFIC_LIGHT_CLASS_IDS = [9, 10, 11, 12, 13, 14, 15, 16]

# Debug class ID values - will print all detected class IDs
DEBUG_CLASS_IDS = True


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class TrafficLightColorDetector:
    def __init__(self):
        # Color thresholds for HSV color space
        self.color_ranges = {
            "red": [
                ((0, 70, 50), (10, 255, 255)),
                ((170, 70, 50), (180, 255, 255))
            ],
            "yellow": [((20, 100, 100), (30, 255, 255))],
            "green": [((40, 50, 50), (80, 255, 255))]
        }
    
    def detect_color(self, image, bbox):
        # Extract traffic light region
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "unknown"
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for each color
        for color, ranges in self.color_ranges.items():
            mask = None
            for lower, upper in ranges:
                if mask is None:
                    mask = cv2.inRange(hsv, lower, upper)
                else:
                    mask = mask | cv2.inRange(hsv, lower, upper)
            
            if mask is None:
                continue
                
            # If enough pixels match the color, return it
            pixel_count = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            if pixel_count > total_pixels * 0.05:  # At least 5% of pixels
                return color
        
        return "unknown"


def main():
    # Load video info
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    
    # Load models
    model = YOLO(r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\models\yolov8y.pt")
    
    # Initialize ByteTrack for tracking
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps
    )
    
    # Manual calculation for thickness and text_scale
    resolution_width, resolution_height = video_info.resolution_wh
    thickness = max(2, int(min(resolution_width, resolution_height) / 400))
    text_scale = max(0.6, min(resolution_width, resolution_height) / 800)
    
    # Color definitions
    colors = {
        'red': (0, 0, 255),       # Red
        'yellow': (0, 255, 255),  # Yellow
        'green': (0, 255, 0),     # Green
        'license': (255, 165, 0), # Orange for license plates
        'unknown': (255, 255, 255), # White for unknown
        'slow': (0, 255, 0),      # Green for slow vehicles
        'medium': (0, 255, 255),  # Yellow for medium speed
        'fast': (0, 0, 255)       # Red for fast vehicles
    }
    
    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_padding=5
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2
    )
    
    # Frame generator
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
    
    # Initialize the polygon zone
    polygon_zone = sv.PolygonZone(
        polygon=SOURCE,
        frame_resolution_wh=video_info.resolution_wh
    )
    
    # Initialize view transformer
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    
    # Initialize traffic light color detector
    traffic_light_detector = TrafficLightColorDetector()
    
    # Tracking data
    tracked_plates = {}
    tracked_lights = {}
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    
    # Frame counter
    frame_count = 0
    
    # Process the video
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame in frame_generator:
            frame_count += 1
            start_time = time.time()
            
            # Run model on frame
            result = model(frame)[0]
            
            # Debug - print all detection info
            if frame_count <= 3:
                print(f"Model output raw classes: {result.boxes.cls}")
                print(f"Model output raw names: {result.names}")
                print(f"Model output detected items: {[f'{result.names[int(c)]}' for c in result.boxes.cls]}")
            
            # Direct access to green-light detections from model result
            green_light_indices = []
            try:
                green_light_indices = [i for i, c in enumerate(result.boxes.cls) if result.names[int(c)] == 'green-light']
            except Exception as e:
                print(f"Error finding green-lights: {e}")
                
            # Standard detections processing    
            detections = sv.Detections.from_ultralytics(result)
            
            # Filter by confidence
            detections = detections[detections.confidence > 0.3]
            
            # Filter by region of interest
            detections = detections[polygon_zone.trigger(detections)]
            
            # Apply NMS
            detections = detections.with_nms(threshold=0.7)
            
            # Update trackers
            detections = byte_track.update_with_detections(detections=detections)
            
            # Get license plate detections
            if len(detections) > 0:
                # Try to get license plate detections
                try:
                    plate_detections = detections[detections.class_id == LICENSE_PLATE_CLASS_ID]
                except Exception as e:
                    print(f"Error filtering plate detections: {e}")
                    plate_detections = sv.Detections.empty()
            else:
                plate_detections = sv.Detections.empty()
                
            # In supervision 0.16.0, we need to manually calculate bottom center points
            if len(detections) > 0:
                xyxy = detections.xyxy
                points = np.zeros((len(xyxy), 2))
                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = box
                    # Bottom center point
                    points[i] = [(x1 + x2) / 2, y2]
                
                points = view_transformer.transform_points(points=points).astype(int)
            else:
                points = np.array([])

            # Update coordinates for speed calculation
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)
            
            # Copy frame for annotations
            annotated_frame = frame.copy()
            
            # Add a title bar
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame, 
                "THE EYE OF GOD - Trafik Ihlal Tespit Sistemi", 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            
            # Draw the region of interest
            pts = SOURCE.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(annotated_frame, [pts], True, (0, 255, 255), 2)
            
            # Process license plates - only if we have valid plate_detections
            plate_labels = []
            if hasattr(plate_detections, 'xyxy') and plate_detections.xyxy is not None and len(plate_detections) > 0:
                for i, (xyxy, confidence, class_id, tracker_id) in enumerate(
                    zip(plate_detections.xyxy, plate_detections.confidence, 
                        plate_detections.class_id, plate_detections.tracker_id)
                ):
                    # Add plate info to tracking
                    tracked_plates[tracker_id] = {
                        'xyxy': xyxy,
                        'confidence': confidence,
                        'frame': frame_count
                    }
                    
                    # Prepare label
                    plate_labels.append(f"Lisence Plate #{tracker_id}")
                    
                    # Draw custom colored box
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                colors['license'], thickness)
                
                # Add annotations for license plates
                annotated_frame = label_annotator.annotate(annotated_frame, plate_detections, plate_labels)
            
            # Process all detections from raw model output to catch traffic lights
            # This bypasses the class ID issue by using the raw detection boxes
            for i, box in enumerate(result.boxes.xyxy):
                try:
                    class_id = int(result.boxes.cls[i].item())
                    class_name = result.names[class_id]
                    confidence = float(result.boxes.conf[i].item())
                    
                    # Only process green-lights or traffic lights
                    if 'light' in class_name and confidence > 0.3:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Determine color based on class name
                        if 'green' in class_name:
                            light_color = "green"
                        elif 'red' in class_name:
                            light_color = "red"
                        elif 'yellow' in class_name:
                            light_color = "yellow"
                        else:
                            light_color = "unknown"
                        
                        # Draw the bounding box with thick lines
                        box_color = colors.get(light_color, colors['unknown'])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                    box_color, thickness+2)  # Very thick lines
                        
                        # Add a filled colored bar for better visibility
                        bar_height = max(30, int((y2 - y1) * 0.2))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y1 + bar_height), 
                                    box_color, -1)  # Filled rectangle
                        
                        # Add label text with class name
                        cv2.putText(
                            annotated_frame,
                            f"{class_name} ({confidence:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            box_color,
                            2
                        )
                except Exception as e:
                    # Skip this detection if there's any error
                    if frame_count <= 5:
                        print(f"Error processing detection {i}: {e}")
                    continue
            
            # Process speed calculation and add labels
            speed_labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    speed_labels.append(f"#{tracker_id}")
                else:
                    start = coordinates[tracker_id][-1]
                    end = coordinates[tracker_id][0]
                    distance = abs(start - end)
                    tracking_time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / tracking_time * 3.6
                    
                    # Choose color based on speed
                    if speed < 30:
                        speed_color = colors['slow']
                    elif speed < 60:
                        speed_color = colors['medium']
                    else:
                        speed_color = colors['fast']
                    
                    # Format speed with symbol
                    speed_labels.append(f"#{tracker_id} - {int(speed)} km/h")
            
            # Add trace annotations and vehicle labels
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, speed_labels)
            
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            
            # Add FPS counter
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", 
                       (annotated_frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add a legend
            legend_height = 230
            legend = np.zeros((legend_height, annotated_frame.shape[1], 3), dtype=np.uint8)
            cv2.rectangle(legend, (0, 0), (legend.shape[1], legend.shape[0]), (30, 30, 30), -1)
            
            # Draw color indicators for traffic lights
            cv2.putText(legend, "Trafik Isigi Renk Kodlari:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(legend, "Kirmizi Isik", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['red'], 2)
            cv2.putText(legend, "Sari Isik", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['yellow'], 2)
            cv2.putText(legend, "Yesil Isik", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['green'], 2)
            cv2.putText(legend, "Plaka", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['license'], 2)
            
            # Draw speed range info
            cv2.putText(legend, "Hiz Araliklari:", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(legend, "< 30 km/s", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['slow'], 2)
            cv2.putText(legend, "30-60 km/s", (300, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['medium'], 2)
            cv2.putText(legend, "> 60 km/s", (550, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['fast'], 2)
            
            # Stats
            stats_x = annotated_frame.shape[1] - 300
            cv2.putText(legend, f"Plaka Sayisi: {len(plate_detections) if plate_detections is not None else 0}", 
                      (stats_x, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['license'], 2)
            
            # Traffic light count from direct detection
            try:
                cv2.putText(legend, f"Trafik Isigi Sayisi: {len(green_light_indices)}", 
                          (stats_x, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error displaying traffic light count: {e}")
            
            # Display with legend
            display_frame = np.vstack((annotated_frame, legend))
            
            # Resize for display if too large
            display_height, display_width = display_frame.shape[:2]
            max_display_height = 900  # Maximum reasonable height
            
            if display_height > max_display_height:
                scale_factor = max_display_height / display_height
                display_width = int(display_width * scale_factor)
                display_height = max_display_height
                display_frame = cv2.resize(display_frame, (display_width, display_height))
            
            # Write frame to output video
            sink.write_frame(annotated_frame)
            
            # Display frame
            cv2.imshow("THE EYE OF GOD - Trafik Ihlal Tespit Sistemi", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()