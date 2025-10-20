import cv2
import numpy as np
import os

# Inisialisasi face recognizer OpenCV dan load model yang sudah dilatih
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer_data.xml")  # Pastikan model telah dilatih sebelumnya

# Folder tempat data wajah disimpan
dataset_folder = "face_recognition_data"

# Mengambil ID dan nama wajah
names = {}
for i, filename in enumerate(os.listdir(dataset_folder)):
    if filename.endswith(".jpg"):
        label = filename.split('_')[0]
        if label not in names:
            names[label] = len(names) + 1

# Inisialisasi face cascade dari OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi utama untuk mengenali wajah
def recognize_face():
    cap = cv2.VideoCapture(1)  # Menggunakan webcam
    

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Mengubah frame ke grayscale untuk pengenalan wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah dengan Haar Cascade
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Potong wajah dari frame
            face_image = gray[y:y+h, x:x+w]

            # Pengenalan wajah: mencocokkan wajah dengan model yang sudah dilatih
            label, confidence = recognizer.predict(face_image)

            # Menampilkan nama yang dikenali
            name = [key for key, value in names.items() if value == label][0]
            confidence_text = f"Confidence: {round(100 - confidence, 2)}%"

            cv2.putText(frame, f"Nama: {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Menampilkan frame video
        cv2.imshow("Face Recognition", frame)

        # Keluar dari program dengan menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Memulai pengenalan wajah
recognize_face()
