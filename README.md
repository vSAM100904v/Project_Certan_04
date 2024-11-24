# AI Identifikasi 75 Jenis Kupu-Kupu Menggunakan CNN

Proyek ini adalah bagian dari tugas kelompok 4 untuk mata kuliah **CERTAN (Kecerdasan Buatan)**. Kami mengembangkan sistem berbasis **Convolutional Neural Network (CNN)** untuk mengidentifikasi **75 jenis kupu-kupu** secara otomatis melalui citra. Tujuan dari proyek ini adalah memanfaatkan teknologi kecerdasan buatan untuk mengenali spesies kupu-kupu secara efisien dan akurat.

---

## 🚀 Fitur Utama
- **Identifikasi Otomatis:** Mampu mengenali 75 jenis kupu-kupu dengan akurasi tinggi.
- **Arsitektur CNN:** Menggunakan model deep learning yang dioptimalkan untuk klasifikasi gambar.
- **Dataset Berkualitas:** Menggunakan dataset yang dikumpulkan dari sumber terpercaya dengan gambar kupu-kupu yang beragam.
- **Antarmuka Sederhana:** Sistem yang mudah digunakan untuk mengunggah gambar dan mendapatkan hasil klasifikasi.

---

## 🛠️ Teknologi yang Digunakan
- **Python 3.8+**
- **TensorFlow / Keras:** Untuk membangun dan melatih model CNN.
- **NumPy & Pandas:** Untuk manipulasi data.
- **Matplotlib & Seaborn:** Untuk visualisasi data dan hasil pelatihan.
- **Flask / Streamlit:** Untuk membangun antarmuka pengguna (opsional).
- **Google Colab / Jupyter Notebook:** Untuk pengembangan dan eksperimen model.

---

## 📁 Struktur Proyek
```
|-- data/
|   |-- train/            # Dataset untuk pelatihan
|   |-- test/             # Dataset untuk pengujian
|
|-- models/
|   |-- cnn_model.h5      # Model yang sudah dilatih
|
|-- notebooks/
|   |-- create_dataset.py # pembuatan data set 
|   |-- Train_cnn.py # Eksplorasi dan preprocessing data
|   |-- predict_cnn.py    # Notebook untuk pelatihan model
|
|-- README.md             # Dokumentasi proyek
|-- requirements.txt      # Daftar library Python yang dibutuhkan
```

---

## 📋 Cara Menjalankan Proyek

### 1. Clone Repository
```
git clone https://github.com/username/repo-nama.git
cd repo-nama
```

### 2. Instalasi Dependensi
```
pip install -r tensorflow
```

### 3. Jalankan Model
Untuk menjalankan model pada gambar yang diunggah, gunakan terminal notebook `python Train_cnn.py` atau aplikasi berbasis Flask/Streamlit:
```
python app/app.py
```
kemudian jalankan file prdiksi  `python predict_cnn.py` lalu pilih file dari directory anda

---

## 🔍 Alur Kerja Model
1. **Preprocessing Data:** Mengubah ukuran gambar, augmentasi, dan normalisasi.
2. **Arsitektur CNN:** Model CNN terdiri dari beberapa lapisan convolutional, pooling, dan fully connected untuk klasifikasi.
3. **Pelatihan Model:** Melatih model menggunakan dataset yang telah dibagi menjadi training dan testing.
4. **Evaluasi:** Mengukur akurasi dan loss model pada data uji.
5. **Prediksi:** Model menerima input gambar baru dan memberikan hasil prediksi jenis kupu-kupu.

---

## 📊 Hasil Akhir
- **Akurasi Pelatihan:** 95% (contoh)
- **Akurasi Pengujian:** 90% (contoh)
- **Loss:** 0.1 (contoh)

Visualisasi pelatihan dan pengujian dapat ditemukan di notebook `Train_cnn.py`.

---

## 🤝 Kontributor
Proyek ini dikerjakan oleh:
- **11423018_Eduward Gilbert Simanjuntak** (Ketua)
- **11423004_Marsel Tambunan** (Anggota)
- **11423016_Samuel Margomgom Tua Sitorus** (Anggota)
- **11423031_Febri Sweeta Br. Lumban Raja** (Anggota)
- **11423059_Venesa Herawaty Hutajulu** (Anggota)

---

## 🌟 Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE). Silakan gunakan dan modifikasi sesuai kebutuhan, dengan memberikan atribusi yang sesuai.

---

### Selamat datang di dunia kecerdasan buatan untuk konservasi! 🌍🦋
