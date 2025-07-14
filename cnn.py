import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Membangun model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # Ganti jumlah output sesuai jumlah kelas
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Gunakan ImageDataGenerator untuk augmentasi data jika perlu
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory('dataset/train', target_size=(100, 100), batch_size=32, class_mode='binary')
validation_generator = datagen.flow_from_directory('dataset/validation', target_size=(100, 100), batch_size=32, class_mode='binary')

model.fit(train_generator, epochs=10, validation_data=validation_generator)

model.summary()
# Menyimpan model setelah pelatihan
model.save('face_recognition_model.h5')

