import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os

# 1. Dataset directory
data_dir = './cheque data'

# 2. Image size and batch
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# 3. Data generator (rescale + split)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 4. Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

# 5. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train
history = model.fit(train_data, validation_data=val_data, epochs=5)

# 7. Save model (optional)
# Option 1 (Recommended Keras format)
model.save('cheque_classifier_model_0.1.keras')


print('✅ Model training complete and saved as cheque_classifier_model_0.1')