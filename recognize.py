import os
import cv2
import json
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Load embeddings & names
def load_embd():
    embd_dict = {}  # Store embeddings
    name_dict = {}  # Store employee names
    embd_path = "embeddings/"
    
    if not os.path.exists(embd_path):
        print(" No embeddings found. Train the model first.")
        return None, None

    for file in os.listdir(embd_path):
        if file.endswith(".npy"):
            emp_id = file.split(".")[0]
            embd_dict[emp_id] = np.load(os.path.join(embd_path, file))

            # Load employee name from JSON
            json_path = os.path.join(embd_path, f"{emp_id}.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    name_dict[emp_id] = data.get("employee_name", "Unknown")
            else:
                name_dict[emp_id] = "Unknown"

    return embd_dict, name_dict  

# Recognize face
def recog_face():
    embd_dict, name_dict = load_embd()
    if not embd_dict:
        return

    detector = MTCNN()
    cap = cv2.VideoCapture(0)  

    print(" Starting face recognition... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Camera is not working.")
            break

        faces = detector.detect_faces(frame)
        if not faces:  
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for face in faces:
            x, y, width, height = face['box']
            face_img = frame[y:y+height, x:x+width]  

            try:
                embedding = DeepFace.represent(face_img, model_name="OpenFace", detector_backend="opencv", enforce_detection=False)

                if isinstance(embedding, list) and len(embedding) > 0:
                    embedding = np.array(embedding[0])
                else:
                    continue

                best_match = None
                best_score = float("inf")

                for emp_id, stored_embds in embd_dict.items():
                    for stored_embd in stored_embds:
                        score = cosine(stored_embd, embedding)
                        if score < best_score:
                            best_score = score
                            best_match = emp_id

                # Display recognized name & ID
                if best_match and best_score < 0.4:
                    emp_name = name_dict.get(best_match, "Unknown")
                    cv2.putText(frame, f"ID: {best_match}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Name: {emp_name}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

            except Exception as e:
                print(f" Error processing face: {e}")

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

    cap.release()  
    cv2.destroyAllWindows()
    print(" Face recognition stopped.")

# Run the recognition function
if __name__ == "__main__":
    recog_face()
