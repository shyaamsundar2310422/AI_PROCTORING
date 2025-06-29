import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Webcam
cap = cv2.VideoCapture(0)

# Landmark sets for each eye
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth
    (150.0, -150.0, -125.0)   # Right mouth
], dtype="double")

def get_eye_image(frame, landmarks, indices):
    h, w = frame.shape[:2]
    eye_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*eye_points)
    x_min, x_max = max(min(x_coords) - 5, 0), min(max(x_coords) + 5, w)
    y_min, y_max = max(min(y_coords) - 5, 0), min(max(y_coords) + 5, h)
    eye_crop = frame[y_min:y_max, x_min:x_max]
    return eye_crop

def detect_eye_direction(eye_img):
    if eye_img.size == 0:
        return "N/A"

    # Skip small regions
    if eye_img.shape[0] < 10 or eye_img.shape[1] < 10:
        return "Too Far"

    # Resize for better visibility
    eye_img = cv2.resize(eye_img, (100, 60), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding for robust pupil detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            w = eye_img.shape[1]
            if cx < w // 3:
                return "Left"
            elif cx > 2 * w // 3:
                return "Right"
            else:
                return "Center"
    return "Undetected"

def get_head_direction(rotation_vector):
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))

    horizontal = "Center"
    if yaw < -15:
        horizontal = "Left"
    elif yaw > 15:
        horizontal = "Right"

    vertical = "Center"
    if pitch < -15:
        vertical = "Up"
    elif pitch > 15:
        vertical = "Down"

    return horizontal, vertical

def draw_label_box(img, lines, x=20, y=20):
    box_w = 320
    line_height = 32
    box_h = line_height * len(lines) + 20

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x + 10, y + 30 + i * line_height),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    display_lines = []

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]
            image_points = np.array([
                (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),
                (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h),
                (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h),
                (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),
                (landmarks.landmark[287].x * w, landmarks.landmark[287].y * h),
                (landmarks.landmark[57].x * w, landmarks.landmark[57].y * h)
            ], dtype="double")

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            face_horizontal, face_vertical = get_head_direction(rot_vec)
            display_lines.append((f"Head Direction: {face_horizontal}, {face_vertical}", (0, 255, 255)))

            left_eye_img = get_eye_image(frame, landmarks, LEFT_EYE_IDX)
            right_eye_img = get_eye_image(frame, landmarks, RIGHT_EYE_IDX)

            left_eye_dir = detect_eye_direction(left_eye_img)
            right_eye_dir = detect_eye_direction(right_eye_img)

            display_lines.append((f"Left Eye: {left_eye_dir}", (255, 0, 255)))
            display_lines.append((f"Right Eye: {right_eye_dir}", (255, 0, 255)))

            # DEBUG preview
            if left_eye_img.size > 0:
                cv2.imshow("Left Eye", cv2.resize(left_eye_img, (150, 100)))
            if right_eye_img.size > 0:
                cv2.imshow("Right Eye", cv2.resize(right_eye_img, (150, 100)))

    draw_label_box(frame, display_lines, x=10, y=10)

    cv2.imshow("Head + Eye Pose Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
