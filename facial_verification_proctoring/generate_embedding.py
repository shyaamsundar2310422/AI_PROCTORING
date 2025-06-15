from deepface import DeepFace
import pickle
import os

def generate_and_store_embedding(student_id):
    os.makedirs("embeddings", exist_ok=True)
    path = f"reference_faces/{student_id}.jpg"
    embedding = DeepFace.represent(img_path=path, model_name="Facenet")[0]['embedding']
    with open(f"embeddings/{student_id}.pkl", "wb") as f:
        pickle.dump(embedding, f)
    print(f"[INFO] Embedding stored for {student_id}")

# Example usage:
generate_and_store_embedding("john_01")
