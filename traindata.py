import cv2
import numpy as np
import os

# Inisialisasi face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Folder tempat data wajah disimpan
dataset_folder = "face_recognition_data"

# Menyimpan gambar wajah dan label
faces = []
labels = []
label_names = {}

# Mengambil semua gambar wajah dan labelnya
for filename in os.listdir(dataset_folder):
    if filename.endswith(".jpg"):
        # Ambil label (nama) dari nama file, contohnya 'name_1.jpg' -> label = 'name'
        label = filename.split('_')[0]
        if label not in label_names:
            label_names[label] = len(label_names) + 1
        label_id = label_names[label]
        
        # Baca gambar wajah
        image_path = os.path.join(dataset_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Tambahkan wajah dan label ke list
        faces.append(image)
        labels.append(label_id)

# Pastikan ada data yang ditemukan untuk melatih
if len(faces) > 0 and len(labels) > 0:
    # Melatih model dengan data wajah
    recognizer.train(faces, np.array(labels))

    # Simpan model yang sudah dilatih
    recognizer.save("face_recognizer_data.xml")
    print("Model pengenalan wajah telah dilatih dan disimpan sebagai 'face_recognizer_data.xml'")
else:
    print("Tidak ada data wajah yang ditemukan untuk dilatih.")
