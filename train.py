import os
import numpy as np
import cv2
from deepface import DeepFace
import json  

def extract_embeddings(employee_id, employee_name):
    dataset_path = f"dataset/{employee_name}_{employee_id}/"  # 🔹 Use same format as register.py
    embedding_path = f"embeddings/{employee_id}.npy"
    metadata_path = f"embeddings/{employee_id}.json"

    if not os.path.exists(dataset_path):
        print(f" No dataset found for Employee ID {employee_id}. Run register.py first.")
        return

    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)  
    embeddings = []

    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue  

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            embedding = DeepFace.represent(img_rgb, model_name="OpenFace", detector_backend="opencv", enforce_detection=False)
            if embedding:
                embeddings.append(np.array(embedding[0]))  
                print(f" Processed {img_name}")

        except:
            continue  

    if embeddings:
        np.save(embedding_path, np.array(embeddings))  
        print(f" Embeddings saved for Employee ID {employee_id}")

        metadata = {"employee_id": employee_id, "employee_name": employee_name}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        print(f" Metadata saved for Employee ID {employee_id}")
    else:
        print(f" No valid embeddings found for Employee ID {employee_id}")

if __name__ == "__main__":
    emp_id = input("Enter Employee ID to train: ")
    emp_name = input("Enter Employee Name: ")
    extract_embeddings(emp_id, emp_name)
