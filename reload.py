import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import tensorflow.keras as keras

tf.__version__
base_dir = "/home/tiina/vegetables/webcam"
IMAGE_SIZE = 224
BATCH_SIZE = 32
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
epochs = 30

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)
'''Prepares training data from directory'''
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training')

'''Prepares validation data from directory'''
val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation')

for image_batch, label_batch in train_generator:
  image_batch.shape, label_batch.shape
  break
image_batch.shape, label_batch.shape

print (train_generator.class_indices)

'''Write data labels to file'''
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt', 'w') as f:
  f.write(labels)

model =tf.keras.models.load_model('try2.h5')

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
es= EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

history = model.fit(train_generator,
                    epochs=epochs, callbacks=[es],
                    validation_data=val_generator)


tf.saved_model.save(model, "retrained/retrained")

converter = tf.lite.TFLiteConverter.from_saved_model("retrained/retrained")
tflite_model = converter.convert()

with open('model_own_retrained.tflite', 'wb') as f:
  f.write(tflite_model)