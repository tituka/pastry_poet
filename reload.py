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
import time
import shutil
print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start_time=time.time()
default_file_name = datetime.datetime.now().strftime("%d%m%H%M")
save_time = datetime.datetime.now().strftime("%d.%m, %H:%M")
parser = argparse.ArgumentParser()
parser.add_argument('--name', default=default_file_name)
parser.add_argument('--retrain', default='/home/tiina/poet/pastry_poet/save/mandatory_save')
parser.add_argument("--from_dir", action="store_true")
parser.add_argument('--source_dir', default="/home/tiina/vegetables/just_webcam")
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--ft_epochs', default=2, type=int)
parser.add_argument('--batch', default=16, type=int)
parser.add_argument('--ft_batch', default=16, type=int)
parser.add_argument('--sp', default=5,  type=int)
parser.add_argument('--ft_sp', default=5, type=int)
parser.add_argument('--s_cond', default='val_loss')
parser.add_argument('--ft_s_cond', default='val_loss')
parser.add_argument("--dlt_all", action="store_true")
loss_function='categorical_crossentropy'

args = parser.parse_args()

DELETE_ALL = args.dlt_all
if DELETE_ALL:
    FILE_NAME= args.name + 'DEL'
else:
    FILE_NAME = args.name
BASE_DIR = args.source_dir
OLD_MODEL=args.retrain
BASE_DIR = args.source_dir
BATCH_SIZE = args.batch
BATCH_SIZE = args.ft_batch
EPOCHS = args.epochs
FT_EPOCHS = args.ft_epochs
STOPPING_CONDITION = args.s_cond
STOPPING_PATIENCE = args.sp
FT_STOPPING_CONDITION = args.ft_s_cond
FT_STOPPING_PATIENCE = args.ft_sp
FROM_DIR=args.from_dir
IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)


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

if FROM_DIR:
    '''Prepares training data from directory'''
    train_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        follow_links=True)

    '''Prepares validation data from directory'''
    val_generator = datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        follow_links=True)

    for image_batch, label_batch in train_generator:
      image_batch.shape, label_batch.shape
      break
    image_batch.shape, label_batch.shape

    print (train_generator.class_indices)

else:
    table=pd.read_csv('t.csv',dtype=str)

    train_generator = datagen.flow_from_dataframe(
        dataframe=table,
        directory='/home/tiina/vegetables/all_in_one/',
        x_col="file name",
        y_col="category",
        has_ext=True, subset="training",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_SIZE, IMAGE_SIZE))

    val_generator = datagen.flow_from_dataframe(
        dataframe=table,
        directory='/home/tiina/vegetables/all_in_one/',
        x_col="file name",
        y_col="category",
        class_mode="categorical",

        has_ext=True, subset="validation",
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_SIZE, IMAGE_SIZE))

'''Write data labels to file'''
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt', 'w') as f:
  f.write(labels)
number_of_categories=len(train_generator.class_indices.keys())

model=tf.keras.models.load_model(OLD_MODEL)

es= EarlyStopping(monitor=STOPPING_CONDITION, mode=STOPPING_MODE, verbose=1, patience=STOPPING_PATIENCE)
bm = ModelCheckpoint("best_models/"+FILE_NAME, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

base_model = model.get_layer('mobilenetv2_1.00_224')
base_model.trainable = False

optimizer=tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=['accuracy'])
base_model.trainable = False
model.summary()

history = model.fit(train_generator,
                    epochs=EPOCHS,
                    callbacks=[es],
                    validation_data=val_generator)
train_time= time.time() - start_time
tools.plot_acc_loss(history, FILE_NAME)
saved_model_dir = 'save/mandatory_save'

tf.saved_model.save(model, saved_model_dir)


base_model.trainable = True

'''Freeze all the layers before the `fine_tune_at` layer'''

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

opt_learning_rate = 1e-5

model.compile(loss=loss_function,
              optimizer = tf.keras.optimizers.Adam(opt_learning_rate),
              metrics=['accuracy'])

model.summary()

ft_es= EarlyStopping(monitor=FT_STOPPING_CONDITION, mode=FT_STOPPING_MODE, verbose=1, patience=FT_STOPPING_PATIENCE)
history_fine = model.fit(train_generator,
                         callbacks=[ft_es, bm],
                         epochs=FT_EPOCHS,
                         validation_data=val_generator)

print('Number of trainable variables = {}'.format(len(model.trainable_variables)))


'''Plotting loss and accuracy'''
tools.plot_acc_loss(history, FILE_NAME + '_optimized')


'''Save optimized model in Keras format for re-training'''

model.save('keras_models/' +FILE_NAME+'_retrained' +'.h5')
tools.convert_to_tf(FILE_NAME+'_retrained', model)
tools.convert_to_tf(FILE_NAME, model)

table=pd.read_pickle('train_table1')
layers_string=', '.join([layer.name for layer in model.layers])

table = table.append(pd.Series([FILE_NAME + '_retrained', save_time, min(history_fine.history['val_loss']),
              max(history_fine.history['val_accuracy']),   history_fine.history['val_loss'][-1],
              history_fine.history['val_accuracy'][-1],  min(history.history['val_loss']),
              max(history.history['val_accuracy']),  history.history['val_loss'][-1], history.history['val_accuracy'][-1],
              BASE_DIR, OLD_MODEL,   EPOCHS,  BATCH_SIZE, STOPPING_CONDITION, STOPPING_PATIENCE,
              str(optimizer), 'default_lr', loss_function,  FT_EPOCHS,  BATCH_SIZE,
              STOPPING_CONDITION,  FT_STOPPING_PATIENCE, str(optimizer),
              str(tf.keras.backend.eval(model.optimizer.lr)), loss_function, layers_string, train_time], index=table.columns ), ignore_index=True)

table.to_pickle('train_table1')
table.to_csv('trained_table.csv')

if DELETE_ALL:
    os.remove('labels.txt')
    shutil.rmtree('save/mandatory_save', ignore_errors=True)
    os.remove('keras_models/' +FILE_NAME+'_retrained.h5')
    shutil.rmtree('best_models/' +FILE_NAME, ignore_errors=True)
val_image_batch, val_label_batch = next(iter(val_generator))
val_image_batch2, val_label_batch2 = next(iter(val_generator))
result_batch = model.predict(val_image_batch)
result_batch2 = model.predict(val_image_batch2)
predicted_ids = np.argmax(result_batch, axis=-1)
predicted_ids2 = np.argmax(result_batch2, axis=-1)
image_batch= val_image_batch
predicted_ids=predicted_ids
label_batch=val_label_batch
print(predicted_ids)
print(label_batch)
print(result_batch2)

