import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import tensorflow.keras as keras
import argparse
import datetime
import tools
import pandas as pd

print(tf.__version__)

parser = argparse.ArgumentParser()
default_file_name = datetime.datetime.now().strftime("%d%m%M%S")
save_time = datetime.datetime.now().strftime("%d.%m, %H:%M")
parser.add_argument('--name', default=default_file_name)
args = parser.parse_args()
file_name= args.name

base_dir = "/home/tiina/vegetables/dataset5"
IMAGE_SIZE = 224
BATCH_SIZE = 32
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
epochs = 100
optimizer_epochs=100

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
number_of_categories=len(train_generator.class_indices.keys())
old_model='keras_models/hunny_less2_retrained.h5'
model=tf.keras.models.load_model(old_model)

#model=tf.keras.models.load_model('keras_models/flower2_3_retrained.h5')
es= EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=2)
base_model = model.get_layer('mobilenetv2_1.00_224')

base_model.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
base_model.trainable = False
model.summary()

es= EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

history = model.fit(train_generator,
                    epochs=epochs,
                    callbacks=[es],
                    validation_data=val_generator)

tools.plot_acc_loss(history, file_name)

base_model.trainable = True

'''Freeze all the layers before the `fine_tune_at` layer'''

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(1e-5),
              metrics=['accuracy'])

model.summary()
post_history = model.fit(train_generator,
                         callbacks=[es],
                         epochs=optimizer_epochs,
                         validation_data=val_generator)

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))


'''Plotting loss and accuracy'''
tools.plot_acc_loss(history, file_name + '_optimized')


'''Save optimized model in Keras format for re-training'''
'''
model.save('keras_models/' +file_name+'_retrained' +'.h5')
tools.convert_to_tf(file_name+'_retrained', model)
tools.convert_to_tf(file_name, model)
'''
table=pd.read_pickle('train_table1')

table = table.append(pd.Series([file_name + '_retrained', save_time, min(history_fine.history['val_loss']),
              max(history_fine.history['val_accuracy']),   history_fine.history['val_loss'][-1],
              history_fine.history['val_accuracy'][-1],  min(history.history['val_loss']),
              max(history.history['val_accuracy']),  history.history['val_loss'][-1], history.history['val_accuracy'][-1],
              base_dir, old_model,   epochs,  BATCH_SIZE, stopping_condition, stopping_patience,
              optimizer, 'default_lr', loss_function,  optimizer_epochs,  BATCH_SIZE,
              stopping_condition,  stopping_patience, optimizer,
              tf.keras.backend.eval(model.optimizer.lr), loss_function], index=table.columns ), ignore_index=True)

table.to_pickle('train_table1')
table.to_csv('trained_table.csv')