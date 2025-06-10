import cv2
import pickle
import numpy as np
from deepface import DeepFace

def verify_identity(student_id, threshold=0.6):
    # Load stored embedding
    with open(f"embeddings/{student_id}.pkl", "rb") as f:
        stored_embedding = pickle.load(f)

    # Start webcam
    cap = cv2.VideoCapture(0)
    print("[INFO] Press SPACE to capture face for verification.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Verify - Press SPACE", frame)
        key = cv2.waitKey(1)

        if key % 256 == 32:  # SPACE pressed
            try:
                result = DeepFace.represent(img_path=frame, model_name="Facenet")[0]
                live_embedding = result['embedding']

                # Cosine similarity
                similarity = np.dot(stored_embedding, live_embedding) / (
                    np.linalg.norm(stored_embedding) * np.linalg.norm(live_embedding)
                )

                print(f"[INFO] Cosine Similarity: {similarity:.4f}")
                if similarity > threshold:
                    print("[✅] Identity Verified!")
                else:
                    print("[❌] Identity Mismatch!")
            except Exception as e:
                print(f"[ERROR] Face not detected or another issue: {e}")
            break

        elif key % 256 == 27:  # ESC pressed
            print("[INFO] Verification cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
verify_identity("john_01")
