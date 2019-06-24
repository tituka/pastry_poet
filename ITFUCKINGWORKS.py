from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np
import tools
import argparse
print(tf.__version__)
import argparse
import datetime
import pandas as pd

default_file_name = datetime.datetime.now().strftime("%d%m%H%M")
save_time = datetime.datetime.now().strftime("%d.%m, %H:%M")
parser = argparse.ArgumentParser()
parser.add_argument('--name', default=default_file_name)
args = parser.parse_args()
file_name= args.name

base_dir = "/home/tiina/vegetables/dataset2"
IMAGE_SIZE = 224
BATCH_SIZE = 32
epochs = 100
optimizer_epochs=100


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

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
number_of_categories=len(train_generator.class_indices.keys())
labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(labels)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

base_model.trainable = False
model = tf.keras.Sequential([
  base_model,
  # tf.keras.layers.Conv2D(32, 3, activation='relu'),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(number_of_categories, activation='softmax')
])
loss_function='categorical_crossentropy'
optimizer=tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=['accuracy'])

model.summary()

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))
stopping_condition='val_loss'
stopping_patience=13
es= EarlyStopping(monitor=stopping_condition, mode='min', verbose=1, patience=stopping_patience)
bm = ModelCheckpoint("best_models/"+file_name, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(train_generator, 
                    epochs=epochs, callbacks=[es],
                    validation_data=val_generator)

acc = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

tools.plot_acc_loss(history, file_name)

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
opt_learning_rate=1e-5
model.compile(loss=loss_function,
              optimizer = tf.keras.optimizers.Adam(opt_learning_rate),
              metrics=['accuracy'])
model.summary()

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

history_fine = model.fit(train_generator, 
                         epochs=optimizer_epochs,
                         callbacks=[es],
                         validation_data=val_generator)

tools.plot_acc_loss(history_fine, file_name+'_optimized')

'''Save optimized model in Keras format for re-training'''
'''
model.save('keras_models/' +file_name+'_retrained' +'.h5')
tools.convert_to_tf(file_name+'_retrained', model)
tools.convert_to_tf(file_name, model)
'''
table=pd.read_pickle('train_table1')

table = table.append(pd.Series([file_name, save_time, min(history_fine.history['val_loss']),
              max(history_fine.history['val_accuracy']),   history_fine.history['val_loss'][-1],
              history_fine.history['val_accuracy'][-1],  min(history.history['val_loss']),
              max(history.history['val_accuracy']),  history.history['val_loss'][-1], history.history['val_accuracy'][-1],
              base_dir, 'new_model',   epochs,  BATCH_SIZE, stopping_condition, stopping_patience,
              optimizer, 'default_lr', loss_function,  optimizer_epochs,  BATCH_SIZE,
              stopping_condition,  stopping_patience, optimizer,
              tf.keras.backend.eval(model.optimizer.lr), loss_function], index=table.columns ), ignore_index=True)

table.to_pickle('train_table1')
table.to_csv('trained_table.csv')