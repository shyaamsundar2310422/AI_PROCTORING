import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Haarcascade for face count
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Webcam
cap = cv2.VideoCapture(0)

# Eye indices (MediaPipe landmark indices)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# 3D model points for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye corner
    (225.0, 170.0, -135.0),    # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype="double")

# Draw overlay box with text
def draw_label_box(img, lines, x=10, y=10):
    box_w = 360
    line_height = 32
    box_h = line_height * len(lines) + 20
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x + 10, y + 30 + i * line_height),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

# Beautified face status box
def draw_status_box(frame, text, color=(0, 255, 255)):
    x, y, w, h = 10, 400, 400, 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x + 15, y + 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)

# Get cropped eye image
def get_eye_image(frame, landmarks, indices):
    h, w = frame.shape[:2]
    eye_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*eye_points)
    x_min, x_max = max(min(x_coords) - 5, 0), min(max(x_coords) + 5, w)
    y_min, y_max = max(min(y_coords) - 5, 0), min(max(y_coords) + 5, h)
    return frame[y_min:y_max, x_min:x_max]

# Determine eye direction from eye image
def detect_eye_direction(eye_img):
    if eye_img.size == 0 or eye_img.shape[0] < 10 or eye_img.shape[1] < 10:
        return "N/A"
    eye_img = cv2.resize(eye_img, (100, 60), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)
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

# Get head direction from rotation vector
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

# =================== MAIN LOOP ====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face count detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    face_count = len(faces)
    if face_count == 0:
        status = "No face detected"
        color = (0, 255, 255)
    elif face_count == 1:
        status = "1 face detected"
        color = (0, 255, 0)
    else:
        status = f"{face_count} faces detected"
        color = (0, 0, 255)
    draw_status_box(frame, status, color)

    # Head pose & eye direction
    results = face_mesh.process(rgb)
    display_lines = []
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]
            image_points = np.array([
                (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),   # Nose
                (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h),  # Chin
                (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h),  # Right eye
                (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),    # Left eye
                (landmarks.landmark[287].x * w, landmarks.landmark[287].y * h),  # Right mouth
                (landmarks.landmark[57].x * w, landmarks.landmark[57].y * h)     # Left mouth
            ], dtype="double")

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            face_horizontal, face_vertical = get_head_direction(rot_vec)
            display_lines.append((f"Head: {face_horizontal}, {face_vertical}", (0, 255, 255)))

            # Eye preview and direction
            left_eye_img = get_eye_image(frame, landmarks, LEFT_EYE_IDX)
            right_eye_img = get_eye_image(frame, landmarks, RIGHT_EYE_IDX)
            left_eye_dir = detect_eye_direction(left_eye_img)
            right_eye_dir = detect_eye_direction(right_eye_img)

            display_lines.append((f"Left Eye: {left_eye_dir}", (255, 0, 255)))
            display_lines.append((f"Right Eye: {right_eye_dir}", (255, 0, 255)))

            # Show eye previews
            if left_eye_img.size > 0:
                cv2.imshow("Left Eye", cv2.resize(left_eye_img, (150, 100)))
            if right_eye_img.size > 0:
                cv2.imshow("Right Eye", cv2.resize(right_eye_img, (150, 100)))

    draw_label_box(frame, display_lines, x=10, y=10)
    cv2.imshow("🛡️ Proctoring Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
