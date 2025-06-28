import cv2
import os

def capture_reference_image(student_id):
    os.makedirs("reference_faces", exist_ok=True)
    cap = cv2.VideoCapture(0)
    print("[INFO] Press SPACE to capture image or ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture - Press SPACE", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC pressed
            print("[INFO] Capture cancelled.")
            break
        elif key % 256 == 32:  # SPACE pressed
            path = f"reference_faces/{student_id}.jpg"
            cv2.imwrite(path, frame)
            print(f"[INFO] Image saved to {path}")
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage:
capture_reference_image("john_01")
