import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time  # Import time module for delay
import json  # Import json for real-time data logging
import math  # Import math module for angle calculations

# Load YOLO model
model = YOLO('/home/furkanslinux/batcave/Orangewood/amazon/Vision/best.pt')

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable depth stream

pipeline.start(config)

# Get camera intrinsics
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # Get depth scale
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

prev_time = time.time()  # Initialize time for delay

# Function to convert pixel distances to real-world distances
def pixel_to_mm(pixel_distance, depth, fx):
    return (pixel_distance * depth) / fx  # Convert pixels to mm

# Function to calculate precise angle using depth data
def calculate_precise_angle(depth_frame, x1, y1, x2, y2):
    depth1 = depth_frame.get_distance(x1, y1) * 1000  # Convert to mm
    depth2 = depth_frame.get_distance(x2, y2) * 1000  # Convert to mm
    if depth1 == 0 or depth2 == 0:
        return 0  # Avoid division by zero
    angle = math.degrees(math.atan2(depth2 - depth1, x2 - x1))  # Calculate inclination angle
    return round(angle, 2)

# JSON file for real-time data logging
json_filename = "detection.json"

def save_to_json(data):
    with open(json_filename, "w") as json_file:
        json.dump(data, json_file, indent=4)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Introduce a delay of 1 second
        if time.time() - prev_time < 1.0:
            continue
        prev_time = time.time()
        
        # Run YOLO detection
        results = model.predict(frame, show=False)

        object_list = []  # Store detected objects for priority sorting
        label_list = []  # Store detected labels

        # Process detections
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                label = model.names[int(cls)].lower()  # Get class label from model

                # Ensure coordinates are within valid range
                x1 = max(0, min(x1, 639))
                y1 = max(0, min(y1, 479))
                x2 = max(0, min(x2, 639))
                y2 = max(0, min(y2, 479))

                # Calculate precise angle
                angle = calculate_precise_angle(depth_frame, x1, y1, x2, y2)

                # Store detected object details for packages and frames
                if label in ["package", "frame"]:
                    z_distance = depth_frame.get_distance(center_x, center_y) * 1000  # Convert to mm
                    width_mm = pixel_to_mm(x2 - x1, z_distance, intrinsics.fx)
                    length_mm = pixel_to_mm(y2 - y1, z_distance, intrinsics.fy)
                    object_list.append((label, center_x, center_y, z_distance, width_mm, length_mm, x1, y1, x2, y2, angle))
                else:
                    print("Label Detected")
                    label_list.append((label, x1, y1, x2, y2))

        # Sort detected objects based on depth (closer objects get higher priority)
        object_list.sort(key=lambda x: x[3])  

        # Clear previous output in serial monitor
        print("\033c", end="")

        # Prepare data for JSON file
        json_data = []

        # Assign priorities and display
        for priority, (label, center_x, center_y, z_distance, width_mm, length_mm, x1, y1, x2, y2, angle) in enumerate(object_list, start=1):
            print(f"{priority}: {label.upper()} → X={center_x} mm, Y={center_y} mm, Z={z_distance:.1f} mm, Size: {width_mm:.1f} mm x {length_mm:.1f} mm, Angle: {angle}°")

            # Draw bounding box
            color = (0, 255, 0) if label == "frame" else (255, 0, 0)  # Green for "frame", Blue for "package"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.drawMarker(frame, (center_x, center_y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)  # Mark center with 'X'

            # Add priority label
            text = f"{label.upper()} - {priority}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Store data in JSON format
            json_data.append({
                "priority": priority,
                "label": label.upper(),
                "center_x": center_x,
                "center_y": center_y,
                "depth": round(z_distance, 1),
                "width_mm": round(width_mm, 1),
                "length_mm": round(length_mm, 1),
                "angle": angle
            })

        # Store detected labels
        for label, x1, y1, x2, y2 in label_list:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            json_data.append({"label": label.upper(), "bounding_box": [x1, y1, x2, y2]})

        # Save data to JSON file
        save_to_json(json_data)

        # Display the detection frame
        cv2.imshow("YOLO Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
