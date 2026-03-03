# Classification NIH Chest X-Rays

## Ringkasan
Proyek ini mengembangkan pipeline *deep learning* untuk analisis citra rontgen dada dari **NIH Chest X-Rays Dataset** dengan dua tujuan utama:
1. **Klasifikasi multi-label** penyakit toraks menggunakan model **CNN** dan **ResNet**.
2. **Deteksi objek** area penyakit menggunakan model **YOLO**.

Notebook utama proyek berada pada file `models.ipynb`.

## Tujuan Penelitian
- Membangun baseline klasifikasi penyakit toraks berbasis citra X-ray.
- Membandingkan performa arsitektur CNN kustom dan ResNet transfer learning.
- Menyusun pipeline konversi anotasi bounding box ke format YOLO untuk tugas deteksi objek.
- Menyediakan keluaran evaluasi yang dapat dianalisis ulang (CSV prediksi, metrik, dan visualisasi).

## Dataset
Sumber data:
- NIH Official: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Kaggle mirror: https://www.kaggle.com/datasets/nih-chest-xrays/data

Karakteristik utama:
- Jumlah citra: sekitar **112.120** gambar X-ray dada.
- Tugas klasifikasi: **multi-label** (satu gambar dapat memiliki lebih dari satu diagnosis).
- Tugas deteksi: menggunakan anotasi bounding box dari `BBox_List_2017.csv`.

## Metodologi
### 1. Pra-pemrosesan Data
- Memuat metadata label dari `Data_Entry_2017.csv`.
- Menstandarkan nama kolom dan tipe data numerik.
- Membentuk *target labels* dengan mengecualikan kelas `No Finding` dari daftar diagnosis utama.
- Melakukan *multi-label encoding* untuk setiap penyakit.
- Menggabungkan metadata label dengan path citra aktual.
- Melakukan *patient-wise split* menjadi train/validation/test untuk mengurangi kebocoran data antar pasien.

### 2. Pipeline Input Citra (TensorFlow)
- Ukuran input: `224 x 224`.
- Citra dibaca lalu dinormalisasi ke rentang `[0, 1]`.
- Citra diproses sebagai grayscale dan dikonversi ke 3 kanal agar kompatibel dengan arsitektur berbasis ImageNet.
- Dataset dibangun menggunakan `tf.data` dengan *shuffle*, *batching*, dan *prefetch*.

### 3. Klasifikasi Multi-Label
#### CNN
- Arsitektur konvolusional bertingkat dengan `Conv2D`, `MaxPool2D`, `GlobalAveragePooling2D`, dan `Dropout`.
- Lapisan output: `Dense(num_classes, activation="sigmoid")`.
- Loss: `binary_crossentropy`.
- Metrik: `BinaryAccuracy`, `ROC-AUC`, `PR-AUC`, `Precision`, `Recall`.

#### ResNet
- Backbone: `ResNet50` pra-latih ImageNet (`include_top=False`).
- Tahap 1: *feature extractor* (backbone dibekukan).
- Tahap 2: *fine-tuning* (sebagian layer akhir backbone dibuka).
- Loss dan metrik sama dengan CNN untuk perbandingan yang konsisten.

### 4. Deteksi Objek (YOLO)
- Membaca anotasi bounding box.
- Menggabungkan anotasi dengan path citra serta metadata dimensi gambar.
- Mengonversi anotasi ke format YOLO (`x_center, y_center, width, height` ter-normalisasi).
- Menyusun struktur folder train/val/test dan membuat `data.yaml`.
- Melatih model YOLOv8 (`yolov8n.pt`) menggunakan pustaka `ultralytics`.

## Evaluasi dan Keluaran
### Klasifikasi (CNN vs ResNet)
Notebook menghasilkan:
- Evaluasi metrik pada data uji (`loss`, `BinaryAccuracy`, `ROC-AUC`, `PR-AUC`, `Precision`, `Recall`).
- Perbandingan distribusi label aktual vs prediksi.
- File CSV hasil prediksi:
  - `prediction_comparison/cnn_label_comparison.csv`
  - `prediction_comparison/resnet_label_comparison.csv`
- Ringkasan perbandingan:
  - tabel metrik CNN vs ResNet,
  - visualisasi metrik,
  - perbandingan jumlah prediksi benar vs salah (berbasis *exact match*).

### Deteksi (YOLO)
Notebook menghasilkan:
- Struktur dataset YOLO siap latih,
- file konfigurasi `data.yaml`,
- metrik validasi YOLO,
- model tersimpan (mis. `YOLOTrainedModel.pt`).

## Struktur Repositori
- `models.ipynb` : notebook end-to-end untuk pra-pemrosesan, pelatihan, evaluasi, dan perbandingan model.
- `LICENSE` : informasi lisensi proyek.

## Cara Menjalankan
### Opsi yang direkomendasikan: Kaggle Notebook
1. Unggah proyek ke Kaggle atau buka notebook pada lingkungan Kaggle.
2. Pastikan dataset NIH Chest X-Rays telah terpasang sebagai input.
3. Aktifkan GPU pada pengaturan *Accelerator*.
4. Jalankan `models.ipynb` dari awal hingga akhir secara berurutan.

### Opsi lokal
1. Siapkan Python 3.10+.
2. Instal dependensi utama: TensorFlow, scikit-learn, pandas, numpy, matplotlib, Pillow, ultralytics.
3. Sesuaikan `datasetPath` dan `bboxPath` pada notebook.
4. Jalankan notebook secara berurutan.

## Catatan Reproduksibilitas
- Gunakan urutan eksekusi sel yang konsisten dari awal notebook.
- Simpan bobot model terbaik (`ModelCheckpoint`) untuk analisis ulang.
- Untuk pelaporan ilmiah, sertakan hasil beberapa kali pelatihan dengan seed berbeda agar estimasi performa lebih stabil.

## Keterbatasan
- Kualitas label pada dataset medis publik dapat mengandung ketidakpastian.
- Ketidakseimbangan kelas penyakit dapat memengaruhi metrik agregat.
- Metrik *exact match* untuk multi-label bersifat ketat dan bisa terlihat rendah meskipun prediksi parsial sudah informatif.

## Pengembangan Lanjutan
- Menambahkan *class weighting* atau *focal loss* untuk kelas tidak seimbang.
- Menambahkan kalibrasi threshold per-label, bukan threshold global tunggal.
- Melakukan evaluasi eksternal pada dataset institusi berbeda untuk uji generalisasi.
- Menambahkan *explainability* (misalnya Grad-CAM) untuk interpretabilitas klinis.
