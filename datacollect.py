import cv2
import mediapipe as mp
import numpy as np
import os

# Inisialisasi MediaPipe untuk deteksi wajah dan pengenalan wajah
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi OpenCV face recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Folder tempat data wajah akan disimpan
dataset_folder = "face_recognition_data"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Fungsi untuk mendeteksi wajah dan menyimpan gambar
def save_face_data(name, face_id, face_image):
    # Simpan gambar wajah dengan ID yang sesuai
    cv2.imwrite(f"{dataset_folder}/{name}_{face_id}.jpg", face_image)

# Fungsi utama
def main():
    cap = cv2.VideoCapture(1)  # Menggunakan webcam
    
    # Variable untuk penyimpanan nama
    name = ""
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            
            # Mengubah warna frame ke RGB untuk digunakan dengan MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_detection.process(rgb_frame)
            
            if result.detections:
                for detection in result.detections:
                    # Gambarkan bounding box wajah
                    mp_drawing.draw_detection(frame, detection)

            # Konversi frame ke grayscale untuk pengenalan wajah
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah dengan OpenCV
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Gambarkan kotak wajah
                face_image = gray[y:y+h, x:x+w]
                
                # Pengenalan wajah: jika wajah baru, simpan data wajahnya
                if len(os.listdir(dataset_folder)) == 0 or not name:  # Jika belum ada data wajah
                    name = input("Masukkan nama untuk wajah ini: ")
                    face_id = len([f for f in os.listdir(dataset_folder) if f.endswith('.jpg')]) + 1  # Hitung ID berdasarkan file .jpg
                    save_face_data(name, face_id, face_image)
                else:
                    print("Wajah ditemukan, apakah ingin menambah wajah baru? (y/n): ")
                    if cv2.waitKey(1) & 0xFF == ord('y'):
                        name = input("Masukkan nama untuk wajah ini: ")
                        face_id = len([f for f in os.listdir(dataset_folder) if f.endswith('.jpg')]) + 1
                        save_face_data(name, face_id, face_image)
            
            # Tampilkan hasil deteksi dan pengenalan wajah
            cv2.imshow("Face Detection", frame)

            # Keluar dari program dengan menekan tombol 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
