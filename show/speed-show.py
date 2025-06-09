from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# --- VİDEO KONUMU ---
SOURCE_VIDEO_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\video-data\test-video.mp4"
TARGET_VIDEO_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\show\speed-show-output\speed-show-output.mp4"

# --- KALİBRASYON NOKTALARI (VİDEOYA GÖRE DÜZENLEMEN GEREK!) ---
SOURCE = np.array([(941, 228), (1284, 219), (1905, 647), (451, 718)])
TARGET_WIDTH = 11
TARGET_HEIGHT = 25
TARGET = np.array(
    [[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]]
)


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


def main():
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    model = YOLO(r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\models\yolov8x.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps
    )

    # Manual calculation for thickness and text_scale since these utilities aren't available in supervision 0.16.0
    resolution_width, resolution_height = video_info.resolution_wh
    thickness = max(2, int(min(resolution_width, resolution_height) / 400))  # Slightly thicker lines
    text_scale = max(0.6, min(resolution_width, resolution_height) / 800)    # Slightly larger text
    
    # Custom color palette for better aesthetics - removed direct color assignments that aren't compatible
    colors = {
        'slow': (0, 255, 0),      # Green for slow vehicles
        'medium': (0, 255, 255),  # Yellow for medium speed
        'fast': (0, 0, 255),      # Red for fast vehicles
    }

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_padding=5
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2
    )

    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
    polygon_zone = sv.PolygonZone(
        polygon=SOURCE,
        frame_resolution_wh=video_info.resolution_wh
    )
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > 0.3]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=0.7)
            detections = byte_track.update_with_detections(detections=detections)

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

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    start = coordinates[tracker_id][-1]
                    end = coordinates[tracker_id][0]
                    distance = abs(start - end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    
                    # Choose color based on speed
                    if speed < 30:
                        speed_color = colors['slow']
                    elif speed < 60:
                        speed_color = colors['medium']
                    else:
                        speed_color = colors['fast']
                    
                    # Format speed with symbol
                    labels.append(f"#{tracker_id} - {int(speed)} km/h")
                    
                    # Box color customization removed as it's not compatible with supervision 0.16.0

            annotated_frame = frame.copy()
            
            # Add a title/info bar
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame, 
                "THE EYE OF GOD - Hiz Tespiti", 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            
            # Draw the region of interest
            pts = SOURCE.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(annotated_frame, [pts], True, (0, 255, 255), 2)
            
            # Add annotators
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

            sink.write_frame(annotated_frame)
            
            # Add a legend
            legend_height = 150
            legend = np.zeros((legend_height, annotated_frame.shape[1], 3), dtype=np.uint8)
            cv2.rectangle(legend, (0, 0), (legend.shape[1], legend.shape[0]), (30, 30, 30), -1)
            
            # Draw speed range info
            cv2.putText(legend, "Hiz Araliklari:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(legend, "< 30 km/s", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['slow'], 2)
            cv2.putText(legend, "30-60 km/s", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['medium'], 2)
            cv2.putText(legend, "> 60 km/s", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['fast'], 2)
            
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
            
            cv2.imshow("Tanri'nin Gozu - Hiz Tespiti", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
