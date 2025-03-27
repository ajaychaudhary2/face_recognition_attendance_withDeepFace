import os
import numpy as np
import cv2
import json
import logging
from deepface import DeepFace

class FaceEmbeddingExtractor:
    def __init__(self, employee_id, employee_name):
        self.employee_id = employee_id
        self.employee_name = employee_name
        self.dataset_path = f"dataset/{employee_name}_{employee_id}/"
        self.embedding_path = f"embeddings/{employee_id}.npy"
        self.metadata_path = f"embeddings/{employee_id}.json"
        self.log_path = "logs/embedding_extraction.log"

        # Ensure required directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)

        # Configure logging
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def extract_embeddings(self):
        if not os.path.exists(self.dataset_path):
            logging.warning(f"No dataset found for Employee ID {self.employee_id}. Run register.py first.")
            return

        embeddings = []
        for img_name in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue  

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                embedding = DeepFace.represent(img_rgb, model_name="OpenFace", detector_backend="opencv", enforce_detection=False)
                if embedding:
                    embeddings.append(np.array(embedding[0]))  
                    logging.info(f"Processed {img_name}")

            except Exception as e:
                logging.error(f"Error processing {img_name}: {str(e)}")
                continue  

        if embeddings:
            np.save(self.embedding_path, np.array(embeddings))
            logging.info(f"Embeddings saved for Employee ID {self.employee_id}")

            metadata = {"employee_id": self.employee_id, "employee_name": self.employee_name}
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f)

            logging.info(f"Metadata saved for Employee ID {self.employee_id}")
        else:
            logging.warning(f"No valid embeddings found for Employee ID {self.employee_id}")

if __name__ == "__main__":
    emp_id = input("Enter Employee ID to train: ")
    emp_name = input("Enter Employee Name: ")
    extractor = FaceEmbeddingExtractor(emp_id, emp_name)
    extractor.extract_embeddings()
