import cv2
import os
from mtcnn import MTCNN

def capture_images(employee_name, employee_id):
    save_path = f"dataset/{employee_name}_{employee_id}/"

    if os.path.exists(save_path):
        print(f" Employee {employee_name} (ID: {employee_id}) already exists.")
        return

    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Camera is not accessible. Please check your webcam.")
        return

    detector = MTCNN()

    print(f"Capturing images for {employee_name} (ID: {employee_id})... Move your face into the frame.")

    count = 0  # Count only valid face images

    while count < 50:
        ret, frame = cap.read()
        if not ret:
            print(" Camera error. Exiting...")
            break

        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, width, height = face['box']

            # Ensure cropping is within image bounds
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y+height, x:x+width]

            if face_img.size == 0:
                continue  # Skip if the cropped image is invalid

            count += 1
            img_path = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)  # Save only the cropped face
            print(f" Image {count} captured (Face detected)")

            # Draw bounding box for visualization only (not saved)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        cv2.imshow("Capturing Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
            print("\n Capture stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 50:
        print(f" Successfully captured 50 face images for {employee_name} (ID: {employee_id})")
    else:
        print(f" Only {count} images were captured. Please try again.")

if __name__ == "__main__":
    emp_name = input("Enter Employee Name: ").strip()
    emp_id = input("Enter Employee ID: ").strip()
    capture_images(emp_name, emp_id)
