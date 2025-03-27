import os
import cv2
import json
import logging
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Configure logging
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_folder, "face_recognition.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class FaceRecognition:
    def __init__(self):
        """Initialize the face recognition system"""
        self.embd_dict, self.name_dict = self.load_embeddings()
        self.detector = MTCNN()
        self.threshold = 0.4  # Cosine similarity threshold for matching
        if not self.embd_dict:
            logging.warning("No embeddings found. Train the model first.")
            print("No embeddings found. Train the model first.")

    def load_embeddings(self):
        """Load stored embeddings and names"""
        embd_dict = {}
        name_dict = {}
        embd_path = "embeddings/"
        
        if not os.path.exists(embd_path):
            logging.warning("Embeddings folder not found.")
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

        logging.info(f"Loaded {len(embd_dict)} employee embeddings.")
        return embd_dict, name_dict  

    def recognize_faces(self):
        """Recognize faces in real-time using webcam"""
        if not self.embd_dict:
            return

        cap = cv2.VideoCapture(0)
        print("Starting face recognition... Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Camera is not working.")
                print("Camera is not working.")
                break

            faces = self.detector.detect_faces(frame)
            if not faces:
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            for face in faces:
                x, y, width, height = face['box']
                face_img = frame[y:y+height, x:x+width]

                try:
                    embedding = DeepFace.represent(
                        face_img, model_name="OpenFace", detector_backend="opencv", enforce_detection=False
                    )

                    if isinstance(embedding, list) and len(embedding) > 0:
                        embedding = np.array(embedding[0])
                    else:
                        continue

                    best_match, best_score = None, float("inf")

                    for emp_id, stored_embds in self.embd_dict.items():
                        for stored_embd in stored_embds:
                            score = cosine(stored_embd, embedding)
                            if score < best_score:
                                best_score, best_match = score, emp_id

                    # Display recognized face
                    if best_match and best_score < self.threshold:
                        emp_name = self.name_dict.get(best_match, "Unknown")
                        cv2.putText(frame, f"ID: {best_match}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Name: {emp_name}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        logging.info(f"Recognized Employee {emp_name} (ID: {best_match}) with confidence {1 - best_score:.2f}")
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

                except Exception as e:
                    logging.error(f"Error processing face: {e}")
                    print(f"Error processing face: {e}")

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Face recognition stopped.")
        print("Face recognition stopped.")

if __name__ == "__main__":
    recognizer = FaceRecognition()
    recognizer.recognize_faces()
