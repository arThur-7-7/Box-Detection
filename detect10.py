import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time  # Import time module for delay

# Load YOLO model
model = YOLO('/home/furkanslinux/batcave/Orangewood/amazon/Vision/best.pt')

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable depth stream

pipeline.start(config)

prev_time = time.time()  # Initialize time for delay

def draw_text_table(table_img, text, row, col_start=10, row_height=30, font_scale=0.5, color=(255, 255, 255)):
    """Helper function to draw text on a separate window for displaying coordinates."""
    y_pos = row * row_height + 30
    cv2.putText(table_img, text, (col_start, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert to numpy array
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Introduce a delay to stabilize fluctuations
        if time.time() - prev_time < 0.2:  # Delay of 200ms
            continue
        prev_time = time.time()
        
        # Run YOLO detection
        results = model.predict(frame, show=False)

        # Create a blank image for the coordinate table
        table_img = np.zeros((400, 600, 3), dtype=np.uint8)

        # Draw table header
        draw_text_table(table_img, "Detected Objects", 0, col_start=200, font_scale=0.7, color=(0, 255, 0))
        draw_text_table(table_img, "Label       X1  Y1  X2  Y2  CenterX  CenterY  Depth(m)", 1, font_scale=0.6, color=(255, 255, 0))

        row = 2  # Start row for objects in table

        # Draw bounding boxes, labels, calculate (X, Y, Z), and mark center (only for 'package')
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                z_distance = depth_frame.get_distance(center_x, center_y)  # Get depth (Z)

                label = model.names[int(cls)]  # Get class label from model
                text = f"{label}: {z_distance:.2f}m"

                # Print coordinates
                print(f"Detected {label} at Bounding Box: ({x1}, {y1}), ({x2}, {y2}), Depth: {z_distance:.2f}m")

                # Add entry to the coordinate table
                draw_text_table(table_img, f"{label:10} {x1:4} {y1:4} {x2:4} {y2:4} {center_x:7} {center_y:7} {z_distance:.2f}", row)
                row += 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Add background for label
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Draw 'X' mark ONLY if the label is "package"
                if label.lower() == "package":
                    print(f"Drawing center 'X' for package at ({center_x}, {center_y}, {z_distance:.2f}m)")
                    cv2.line(frame, (center_x - 5, center_y - 5), (center_x + 5, center_y + 5), (0, 0, 255), 2)
                    cv2.line(frame, (center_x - 5, center_y + 5), (center_x + 5, center_y - 5), (0, 0, 255), 2)

        # Display the detection frame
        cv2.imshow("YOLO Detection", frame)

        # Display the coordinates table
        cv2.imshow("Object Coordinates", table_img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
