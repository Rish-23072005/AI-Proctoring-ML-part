import cv2
import mediapipe as mp
from deepface import DeepFace
from gaze_tracking import GazeTracking
import time
import torch
import torchvision.transforms as T
from torchvision import models

# Initialize Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Gaze Tracking
gaze = GazeTracking()

# Initialize Object Detection Model (YOLOv5)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
transform = T.Compose([T.ToTensor()])

# Alert System
class AlertSystem:
    def __init__(self):
        self.alert_count = 0

    def trigger_alert(self, message):
        self.alert_count += 1
        print(f"[ALERT] {message} - Count: {self.alert_count}")

alert_system = AlertSystem()

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Face Recognition Parameters
face_verification_threshold = 0.65
focus_threshold = 5  # 5 seconds of focus deviation triggers an alert
last_focus_time = time.time()

# Object Detection Labels
OBJECT_CLASSES = ["phone", "laptop", "tablet"]

def detect_objects(frame):
    img_tensor = transform(frame).unsqueeze(0)
    detections = model(img_tensor)[0]
    for i, score in enumerate(detections['scores']):
        if score > 0.6:
            label = detections['labels'][i].item()
            if label in OBJECT_CLASSES:
                alert_system.trigger_alert(f"{OBJECT_CLASSES[label-1]} Detected")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Face Recognition
    try:
        result = DeepFace.verify(frame, db_path="path_to_face_db")
        if not result['verified']:
            alert_system.trigger_alert("Unrecognized Face Detected")
    except Exception:
        alert_system.trigger_alert("Face Not Detected")

    # Body Posture Detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        cv2.putText(frame, "Posture Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        alert_system.trigger_alert("Unusual Posture Detected")

    # Gaze Tracking
    gaze.refresh(frame)
    if gaze.is_right() or gaze.is_left():
        if time.time() - last_focus_time > focus_threshold:
            alert_system.trigger_alert("Prolonged Distraction Detected")
    else:
        last_focus_time = time.time()

    # Object Detection
    detect_objects(frame)

    # Display the frame
    cv2.imshow("AI Proctoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
