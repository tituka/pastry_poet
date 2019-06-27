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
import shutil


default_file_name = datetime.datetime.now().strftime("%d%m%H%M")
save_time = datetime.datetime.now().strftime("%d.%m, %H:%M")
parser = argparse.ArgumentParser()
parser.add_argument('--name', default=default_file_name)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--ft_epochs', default=2, type=int)
parser.add_argument('--sp', default=5,  type=int)
parser.add_argument('--ft_sp', default=5, type=int)
parser.add_argument('--s_cond', default='val_loss')
parser.add_argument('--ft_s_cond', default='val_loss')
parser.add_argument('--delete_all', default=False, type=bool)




args = parser.parse_args()
DELETE_ALL = args.delete_all
if DELETE_ALL:
    FILE_NAME= args.name + 'DEL'
else:
    FILE_NAME = args.name
BASE_DIR = "/home/tiina/vegetables/withpaper"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = args.epochs
FT_EPOCHS = args.ft_epochs
STOPPING_CONDITION = args.s_cond
STOPPING_PATIENCE = args.sp
FT_STOPPING_CONDITION = args.ft_s_cond
FT_STOPPING_PATIENCE = args.ft_sp


if STOPPING_CONDITION=='val_loss':
    STOPPING_MODE='min'
else:
    STOPPING_MODE='max'

if FT_STOPPING_CONDITION == 'val_loss':
    FT_STOPPING_MODE = 'min'
else:
    FT_STOPPING_MODE = 'max'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory(
    BASE_DIR,
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

'''Create the base model from the pre-trained model MobileNet V2'''
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

es= EarlyStopping(monitor=STOPPING_CONDITION, mode=STOPPING_MODE, verbose=1, patience=STOPPING_PATIENCE)
bm = ModelCheckpoint("best_models/"+FILE_NAME, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(train_generator, 
                    epochs=EPOCHS, callbacks=[es],
                    validation_data=val_generator)

acc = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

saved_model_dir = 'save/mandatory_save'

tf.saved_model.save(model, saved_model_dir)

tools.plot_acc_loss(history, FILE_NAME, DELETE_ALL)

base_model.trainable = True

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

ft_es= EarlyStopping(monitor=STOPPING_CONDITION, mode=FT_STOPPING_MODE, verbose=1, patience=FT_STOPPING_PATIENCE)
history_fine = model.fit(train_generator, 
                         epochs=FT_EPOCHS,
                         callbacks=[ft_es],
                         validation_data=val_generator)

tools.plot_acc_loss(history_fine, FILE_NAME+'_optimized', DELETE_ALL)

'''Save optimized model in Keras format for re-training'''

model.save('keras_models/' +FILE_NAME+'.h5')

tools.convert_to_tf(FILE_NAME, model, DELETE_ALL)

table=pd.read_pickle('train_table1')
layers_string=', '.join([layer.name for layer in model.layers])
table=pd.read_pickle('train_table1')
table = table.append(pd.Series([FILE_NAME, save_time, min(history_fine.history['val_loss']),
              max(history_fine.history['val_accuracy']),   history_fine.history['val_loss'][-1],
              history_fine.history['val_accuracy'][-1],  min(history.history['val_loss']),
              max(history.history['val_accuracy']),  history.history['val_loss'][-1], history.history['val_accuracy'][-1],
              BASE_DIR, 'new_model',   EPOCHS,  BATCH_SIZE, STOPPING_CONDITION, STOPPING_PATIENCE,
              optimizer, 'default_lr', loss_function,  FT_EPOCHS,  BATCH_SIZE,
              FT_STOPPING_CONDITION,  FT_STOPPING_PATIENCE, optimizer,
              tf.keras.backend.eval(model.optimizer.lr), loss_function, layers_string], index=table.columns ), ignore_index=True)

table.to_pickle('train_table1')
table.to_csv('trained_table.csv')

if DELETE_ALL:
    os.remove('labels.txt')
    shutil.rmtree('save/mandatory_save', ignore_errors=True)
    os.remove('keras_models/' +FILE_NAME+'.h5')

