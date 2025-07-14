# import cv2
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Load model CNN yang sudah dilatih
# model = tf.keras.models.load_model('face_recognition_model.h5')

# # Inisialisasi Haar Cascade Classifier untuk deteksi wajah
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Mapping dari label numerik ke nama kelas
# class_labels = {0: 'mobil', 1: 'kursi', 2: 'meja', 3: 'bazoka', 4: 'tank'}

# # Ambang batas untuk menentukan apakah prediksi valid
# THRESHOLD = 0.5

# # Menggunakan kamera
# cap = cv2.VideoCapture(0)

# while True:
#     # Baca frame dari kamera
#     ret, frame = cap.read()
    
#     # Konversi frame menjadi grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Deteksi wajah di dalam gambar
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
#     for (x, y, w, h) in faces:
#         # Gambar persegi panjang di sekitar wajah yang terdeteksi
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
#         # Potong wajah dari frame
#         face = frame[y:y + h, x:x + w]
        
#         # Ubah ukuran wajah menjadi ukuran yang sesuai untuk input model
#         face_resized = cv2.resize(face, (100, 100))
        
#         # Persiapkan input untuk model
#         face_array = image.img_to_array(face_resized) / 255.0  # Normalisasi
#         face_array = np.expand_dims(face_array, axis=0)  # Menambahkan dimensi batch
        
#         # Prediksi kelas wajah
#         prediction = model.predict(face_array)

#         # Dapatkan indeks kelas dengan probabilitas tertinggi dan nilainya
#         predicted_class_index = np.argmax(prediction)
#         predicted_class_prob = prediction[0][predicted_class_index]

#         # Tentukan label berdasarkan probabilitas
#         if predicted_class_prob >= THRESHOLD:
#             label = class_labels.get(predicted_class_index, "Unknown")
#         else:
#             label = "Obyek Tidak Dikenal"  # Label untuk wajah yang tidak dikenali
        
#         # Tampilkan label prediksi di sekitar wajah
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     # Tampilkan frame dengan deteksi wajah dan label
#     cv2.imshow('Real-Time Face Recognition', frame)
    
#     # Keluar jika tombol 'q' ditekan
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Tutup kamera dan jendela OpenCV
# cap.release()
# cv2.destroyAllWindows()
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('face_recognition_model.h5')

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Label mapping
class_labels = {0: 'mobil', 1: 'kursi', 2: 'meja', 3: 'bazoka', 4: 'tank'}

# Kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100, 100))

        # Normalize + expand dims
        face_array = image.img_to_array(face_resized) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # Prediksi
        prediction = model.predict(face_array, verbose=0)
        predicted_class_index = np.argmax(prediction)
        confidence = prediction[0][predicted_class_index]

        label = f"{class_labels[predicted_class_index]} ({confidence:.2f})"

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-Time Object Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
