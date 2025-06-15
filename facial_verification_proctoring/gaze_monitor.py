import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh and Drawing Utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (MediaPipe works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # If face landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks on the face
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Get the points for eyes
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[133]
            nose = face_landmarks.landmark[1]

            # Calculate the direction of the gaze by comparing eye and nose positions
            # Simple approach: looking left, right, or center based on eye landmarks
            gaze_direction = "Looking Center"
            if left_eye.x < nose.x and right_eye.x < nose.x:
                gaze_direction = "Looking Left"
            elif left_eye.x > nose.x and right_eye.x > nose.x:
                gaze_direction = "Looking Right"

            # Put gaze direction text on frame
            cv2.putText(frame, gaze_direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Gaze and Head Pose Monitor', frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
