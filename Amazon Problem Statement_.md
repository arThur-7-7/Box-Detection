**Amazon Problem Statement:**

In warehouse logistics, parcels arrive randomly stacked on conveyor belts, requiring manual adjustment to ensure shipping labels face upwards for scanning. This process is labor-intensive and inefficient, especially when parcels overlap or obscure each other. 

This project aims to develop an AI-powered robotic arm system that uses computer vision and depth sensing to detect, segment, and determine the optimal pick-point for each parcel. A robotic arm with a smart gripper will then pick, rotate, and place parcels correctly. By automating parcel orientation and sorting, this system will enhance efficiency, accuracy, and scalability in warehouse automation. 

**Problem:** [LINK](https://drive.google.com/drive/folders/1xAOwa_q-3V6M6ZlJaFpVwxMKN1QacRdK?usp=sharing)

**Roboflow Model:**  [https://app.roboflow.com/furkan-raju7/box-n-envelope/3](https://app.roboflow.com/furkan-raju7/box-n-envelope/3)

**Approach:**

1\. Vision System for Parcel Detection & Pick-Point Estimation

To handle randomly stacked parcels, a multi-camera system (RGB-D \+ overhead LiDAR or stereo vision) is deployed to: here intel real sense.

* Identify individual parcels, even if they are overlapping  
* Determine depth and height variations to find the topmost parcel  
* Locate the shipping label and calculate its orientation

#### AI Model for Parcel Segmentation & Label Detection

* YOLOv8-seg \+ Depth Processing to segment each parcel.  
* Apply a pick-point estimation algorithm to detect the most accessible surface.  
* If a parcel is fully covered, mark it as temporarily inaccessible until the top parcels are processed.

Output from Vision System:

* Parcel Bounding Boxes (XY Position \+ Depth)  
* Pick Point Coordinates (Where to grasp)  
* Label Location & Rotation Needed (0°, 90°, 180°, or 270°)

Model Output:

* Label present? (Yes/No)  
* Label's current orientation? (Using bounding box coordinates)  
* Rotation needed? (Yes → Send to the robotic arm)

---

### 2\. Robotic Arm with Smart Gripper for Parcel Manipulation

Once a pick point is determined, a 6+ DOF robotic arm with a versatile gripper picks up the parcel and reorients it as needed.

#### Choosing the Right Gripper:

* Suction Gripper (Vacuum-Based) for flat surfaces & sealed boxes  
* Adaptive Two-Finger Gripper for irregular shapes

#### Steps for Parcel Orientation:

1. Pick the topmost parcel using the detected pick-point.  
2. Determine the rotation required:  
   * If the label is already facing up → Skip reorientation  
   * If label is facing down → Flip 180°  
   * If label is on the side → Rotate accordingly (90° or 270°)  
3. Adjust parcel position using inverse kinematics (IK).  
4. Place the reoriented parcel back on the conveyor.

---

### 3\. Conveyor & Sorting Mechanism

* The conveyor system is synchronized with the robotic arm to pause momentarily for picking and placing.  
* Parcels are placed back in the correct orientation, ensuring seamless downstream processing.

---

## Technology Stack & Implementation

### Hardware Requirements:

* Vision System: Intel RealSense / ZED 2 Camera \+ Overhead LiDAR  
* Robotic Arm: UR5 / ABB / Custom 6-DOF Arm  
* Gripper: Suction \+ Adaptive Gripper Hybrid  
* Conveyor System: Speed-Controlled with Feedback Sensors

### Software Stack:

* AI Model: YOLOv8 \+ Mask R-CNN (Parcel Segmentation & Label Detection)  
* Depth Processing: OpenCV \+ PCL (Point Cloud Library)  
* Robotic Arm Control: MoveIt\! (ROS2) \+ TensorRT (for fast inference)

Testing using untrained video: [Link](https://drive.google.com/file/d/1f0CpE9X1WJJLfs-XInOfjAi4iOBpxKED/view?usp=sharing) (Amazon Warehouse)

Pictures: [Amazon Wa](https://drive.google.com/file/d/1mBXi_xI39c0eI1gNyIBVtHIXJuuDMnQM/view?usp=sharing) , [PICTURE1](https://drive.google.com/file/d/1SkNBiaqPDhp7OecGyzYo9nI7m6jIJnUp/view?usp=sharing) , [PICTURE2](https://drive.google.com/file/d/1XXBP1dFBSDeA72G_ArV380mtgjBnL1_-/view?usp=sharing) , [PICTURE3](https://drive.google.com/file/d/1hwyUDv3Q3nq5CnGyAjzADJDtFKsoJXFM/view?usp=sharing)

