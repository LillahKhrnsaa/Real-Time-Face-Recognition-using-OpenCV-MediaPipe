
# 🧑‍💻 Real-Time Face Recognition using OpenCV & MediaPipe

Proyek ini adalah sistem **pengenalan wajah *real-time*** menggunakan kombinasi berbagai pustaka untuk deteksi, koleksi data, dan pelatihan model.

Sistem ini menggunakan alur tiga langkah:
1.  **Koleksi Data Wajah:** Menggunakan **MediaPipe** (deteksi) dan **OpenCV Haar Cascade** (cropping) untuk mengambil gambar wajah dari *webcam* dan menyimpannya.
2.  **Pelatihan Model:** Menggunakan algoritma **LBPH Face Recognizer** dari **OpenCV Contrib** untuk melatih model berdasarkan data wajah yang dikumpulkan.
3.  **Pengenalan Wajah:** Melakukan prediksi *real-time* di *webcam* menggunakan model yang sudah dilatih.

---

## 🧠 Teknologi yang Digunakan

| Teknologi | Deskripsi |
| :--- | :--- |
| 🐍 **Python 3.10+** | Bahasa utama untuk pengembangan proyek. |
| 🎥 **OpenCV** | Pustaka utama untuk pemrosesan gambar, akses *webcam*, deteksi wajah (Haar Cascade), dan pengenalan wajah (LBPH). |
| ✋ **MediaPipe** | Digunakan untuk deteksi wajah awal yang cepat saat proses koleksi data. |
| 📊 **NumPy** | Untuk operasi array dan pemrosesan numerik. |
| 📁 **OS** | Untuk manajemen direktori dan file. |

---

## ⚙️ Struktur Folder

```

face-recognition-project/
│
├── collect\_data.py             \# Script untuk mengumpulkan data wajah (input nama & ambil gambar)
├── train\_model.py              \# Script untuk melatih LBPH model
├── recognize\_face.py           \# Script untuk pengenalan wajah real-time
│
├── face\_recognition\_data/      \# Folder tempat gambar wajah mentah disimpan (Otomatis dibuat)
│   ├── [Nama]\_1.jpg
│   ├── [Nama]\_2.jpg
│   └── ...
│
├── face\_recognizer\_data.xml    \# Model hasil pelatihan (Otomatis dibuat setelah training)
├── requirements.txt            \# Daftar dependensi
└── README.md

````

---

## 🚀 Cara Menjalankan Project

### 1️⃣ Persiapan Lingkungan

```bash
# Clone atau unduh proyek
git clone [https://github.com/username/face-recognition-project.git](https://github.com/username/face-recognition-project.git)
cd face-recognition-project

# Buat dan aktifkan virtual environment (venv)
python -m venv .venv
# Aktifkan sesuai OS Anda...

# Install Dependencies
pip install -r requirements.txt
````

Isi dari `requirements.txt`:

```
opencv-python
mediapipe
numpy
```

### 2️⃣ Fase 1: Koleksi Data Wajah

Jalankan skrip ini untuk mengambil gambar wajah Anda dari *webcam*.

```bash
python collect_data.py
```

> 💡 **Instruksi:**
>
> 1.  Ketika wajah terdeteksi, terminal akan meminta Anda **memasukkan nama**.
> 2.  Program akan otomatis mengambil dan menyimpan beberapa *frame* wajah di folder `face_recognition_data/` dengan format nama: `[Nama]_[ID].jpg`.
> 3.  Tekan **`q`** untuk keluar dari mode koleksi data.

### 3️⃣ Fase 2: Pelatihan Model

Setelah mengumpulkan minimal satu set data wajah, latih model pengenalan:

```bash
python train_model.py
```

**Output:**

  - Model akan dilatih menggunakan LBPH (Local Binary Pattern Histograms).
  - Model yang terlatih akan disimpan sebagai **`face_recognizer_data.xml`**.

### 4️⃣ Fase 3: Pengenalan Wajah Real-Time

Jalankan skrip ini untuk memulai pengenalan wajah menggunakan model yang sudah dilatih.

```bash
python recognize_face.py
```

> 🎥 Program akan membuka *webcam* dan menampilkan:
>
>   - Kotak di sekitar wajah.
>   - **Nama** wajah yang dikenali.
>   - **Tingkat Kepercayaan** (Confidence) dari prediksi.

> 💡 **Keluar:** Tekan tombol **`q`** di keyboard untuk menutup jendela *webcam*.

-----

## 💡 Catatan Penting

  - **ID Kamera:** Skrip menggunakan `cv2.VideoCapture(1)`. Jika kamera Anda tidak terbuka, ubah angka tersebut menjadi `0` atau angka lain yang sesuai dengan ID *webcam* Anda.
  - **LBPH Model:** Algoritma **LBPH Face Recognizer** harus dilatih **setiap kali** Anda menambahkan data wajah baru.
  - **Confidence Score:** Nilai *Confidence* yang rendah menunjukkan kecocokan yang tinggi (semakin kecil, semakin baik/yakin).

-----

## 👩‍💻 Author

**Lillah Khairunisa**

  - 📧 `lillahkhairunisa02@gmail.com`
  - 💻 [github.com/LillahKhrnsaa](https://www.google.com/search?q=https://github.com/LillahKhrnsaa)

<!-- end list -->

```
```
