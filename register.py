import cv2
import os
import logging
from mtcnn import MTCNN

# Ensure the logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "face_register.log"),  # Save log in 'logs/' folder
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FaceRegister:
    def __init__(self, employee_name, employee_id):
        self.employee_name = employee_name
        self.employee_id = employee_id
        self.save_path = f"dataset/{employee_name}_{employee_id}/"
        self.detector = MTCNN()
        self.cap = cv2.VideoCapture(0)

        # Ensure camera is accessible
        if not self.cap.isOpened():
            logging.error("Camera is not accessible.")
            raise RuntimeError("Camera is not accessible.")

        # Create dataset directory if not exists
        if os.path.exists(self.save_path):
            logging.warning(f"Employee {employee_name} (ID: {employee_id}) already exists.")
            raise FileExistsError(f"Employee {employee_name} (ID: {employee_id}) already exists.")

        os.makedirs(self.save_path, exist_ok=True)

    def capture_images(self, num_images=50):
        logging.info(f"Starting image capture for {self.employee_name} (ID: {self.employee_id})")
        print(f"Capturing images for {self.employee_name} (ID: {self.employee_id})... Move your face into the frame.")

        count = 0  # Count valid face images

        while count < num_images:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Camera error. Exiting...")
                break

            faces = self.detector.detect_faces(frame)

            for face in faces:
                x, y, width, height = face['box']
                x, y = max(0, x), max(0, y)
                face_img = frame[y:y+height, x:x+width]

                if face_img.size == 0:
                    continue  # Skip invalid cropped image

                count += 1
                img_path = os.path.join(self.save_path, f"{count}.jpg")
                cv2.imwrite(img_path, face_img)
                logging.info(f"Image {count} captured successfully.")
                print(f"Image {count} captured (Face detected)")

                # Draw bounding box for visualization only
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            cv2.imshow("Capturing Image", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Capture stopped by user.")
                print("\nCapture stopped by user.")
                break

        self.cap.release()
        cv2.destroyAllWindows()

        if count == num_images:
            logging.info(f"Successfully captured {num_images} images for {self.employee_name} (ID: {self.employee_id})")
            print(f"Successfully captured {num_images} face images.")
        else:
            logging.warning(f"Only {count} images were captured.")
            print(f"Only {count} images were captured. Please try again.")

if __name__ == "__main__":
    emp_name = input("Enter Employee Name: ").strip()
    emp_id = input("Enter Employee ID: ").strip()
    
    try:
        register = FaceRegister(emp_name, emp_id)
        register.capture_images()
    except (RuntimeError, FileExistsError) as e:
        print(str(e))
