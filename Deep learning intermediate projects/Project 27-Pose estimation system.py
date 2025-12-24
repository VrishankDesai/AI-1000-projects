# Install if not already: pip install mediapipe opencv-python
 
import cv2
import mediapipe as mp
 
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
 
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
 
# Capture video from webcam
cap = cv2.VideoCapture(0)
 
print("📷 Starting real-time pose estimation... Press 'q' to quit.")
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # Convert BGR to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose_estimator.process(image)
 
    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # Draw landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 
    cv2.imshow('Pose Estimation', image)
 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()