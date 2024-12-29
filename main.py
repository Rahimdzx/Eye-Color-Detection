import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Adjust based on your needs
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Define eye landmarks (Mediapipe's face mesh has 468 landmarks)
LEFT_EYE_INDICES = [33, 133, 160, 144, 158, 153]  # Left eye landmarks
RIGHT_EYE_INDICES = [362, 263, 387, 373, 385, 380]  # Right eye landmarks

# Function to detect eye color using K-Means clustering
def detect_eye_color(eye_region):
    # Reshape the eye region to a 2D array of pixels
    pixels = eye_region.reshape(-1, 3)

    # Use K-Means clustering to find dominant colors
    kmeans = KMeans(n_clusters=2)  # Adjust the number of clusters as needed
    kmeans.fit(pixels)

    # Get the dominant color (cluster center)
    dominant_color = kmeans.cluster_centers_.astype(int)[0]

    # Map the dominant color to a color name
    if dominant_color[2] < 50 and dominant_color[1] < 50 and dominant_color[0] < 50:
        return "Black"
    elif dominant_color[2] < 100 and dominant_color[1] < 100 and dominant_color[0] < 100:
        return "Brown"
    elif dominant_color[0] > dominant_color[1] and dominant_color[0] > dominant_color[2]:
        return "Blue"
    elif dominant_color[1] > dominant_color[0] and dominant_color[1] > dominant_color[2]:
        return "Green"
    else:
        return "Unknown"

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract left and right eye landmarks
            left_eye = [
                (int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0]))
                for i in LEFT_EYE_INDICES
            ]
            right_eye = [
                (int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0]))
                for i in RIGHT_EYE_INDICES
            ]

            # Draw eye landmarks on the frame
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Crop eye regions for color detection
            left_eye_region = frame[
                min(left_eye, key=lambda x: x[1])[1]:max(left_eye, key=lambda x: x[1])[1],
                min(left_eye, key=lambda x: x[0])[0]:max(left_eye, key=lambda x: x[0])[0],
            ]
            right_eye_region = frame[
                min(right_eye, key=lambda x: x[1])[1]:max(right_eye, key=lambda x: x[1])[1],
                min(right_eye, key=lambda x: x[0])[0]:max(right_eye, key=lambda x: x[0])[0],
            ]

            # Detect eye color
            left_eye_color = detect_eye_color(left_eye_region)
            right_eye_color = detect_eye_color(right_eye_region)

            # Display eye color on the frame
            cv2.putText(frame, f"Left: {left_eye_color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Right: {right_eye_color}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Eye Color Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()