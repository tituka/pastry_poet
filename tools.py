import datetime
import time
import datetime
import matplotlib
import pandas as pd
import sys
matplotlib.use('Agg')
import matplotlib.pylab as plt
import shutil
import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_acc_loss(mhistory, mname, delete_all=False):
    acc = mhistory.history['accuracy']
    val_acc = mhistory.history['val_accuracy']
    loss = mhistory.history['loss']
    val_loss = mhistory.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.autoscale(enable=True, axis='y')

    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    # plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    plt.show()
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.autoscale(enable=True, axis='y')

    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    # plt.ylim([0,15.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('graphs/' +'al_'+mname+ '.png')
    if delete_all:
        os.remove('graphs/' +'al_'+mname+ '.png')

def make_sub_dir(name, into):
    path =  into +'/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def convert_to_tf(save_file_name, model, delete_pb_model=True, save_converted=True, delete_all=False):
    saved_model_dir = 'pb_for_tf_models/' + save_file_name
    tf.saved_model.save(model, saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    if save_converted:
        with open('tflite_models/'+ save_file_name+'.tflite', 'wb') as f:
          f.write(tflite_model)
    if delete_pb_model:
        shutil.rmtree('pb_for_tf_models/' + save_file_name)
    if delete_all:
        os.remove('tflite_models/'+ save_file_name+'.tflite')
    return tflite_model

def train(dataframe, image_dir, category, name=None, epochs=3, ft_epochs=3, patience=2, ft_patience=2, train_categories=False, old_dir=None):
    start_time=time.time()
    if name==None:
        FILE_NAME = datetime.datetime.now().strftime("%d%m%H%M")
    else:
        FILE_NAME=name
    IMAGE_SIZE = 224
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    EPOCHS=epochs
    FT_EPOCHS=ft_epochs
    STOPPING_CONDITION = 'val_loss'
    STOPPING_PATIENCE = patience
    FT_STOPPING_CONDITION = 'val_loss'
    FT_STOPPING_PATIENCE = ft_patience
    BATCH_SIZE=16
    loss_function = 'categorical_crossentropy'

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2)

    table = dataframe
    if train_categories:
        column='category'
    else:
        column='product'

    train_generator = datagen.flow_from_dataframe(
        dataframe=table,
        directory=image_dir,
        x_col="file name",
        y_col=column,
        has_ext=True, subset="training",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_SIZE, IMAGE_SIZE))

    val_generator = datagen.flow_from_dataframe(
        dataframe=table,
        directory=image_dir,
        x_col="file name",
        y_col=column,
        class_mode="categorical",
        has_ext=True, subset="validation",
        batch_size=BATCH_SIZE,
        target_size=(IMAGE_SIZE, IMAGE_SIZE))

    for image_batch, label_batch in train_generator:
        image_batch.shape, label_batch.shape
        break
    image_batch.shape, label_batch.shape

    print(train_generator.class_indices)

    classes = train_generator.class_indices.keys()
    labels = '\n'.join(sorted(classes))
    number_of_categories = len(classes)
    misc_path='products_in_categories/'+ category

    with open(make_sub_dir(category, 'misc')+ 'labels.txt', 'w') as f:
        for cls in train_generator.class_indices.keys():
            f.write(cls+'\n')

    optimizer = tf.keras.optimizers.Adam()

    if old_dir==None:
        '''Create the base model from the pre-trained model MobileNet V2'''
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            #tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(number_of_categories, activation='softmax')
        ])
    else:
        model = tf.keras.models.load_model(old_dir)
        base_model = model.get_layer('mobilenetv2_1.00_224')
        base_model.trainable = False

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])

    model.summary()
    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    from tensorflow_core.python.keras.callbacks import ModelCheckpoint, EarlyStopping

    es = EarlyStopping(monitor=STOPPING_CONDITION, mode='min', verbose=1, patience=STOPPING_PATIENCE)
    bm = ModelCheckpoint(make_sub_dir(category, 'best_models')+'first_phase', monitor='val_accuracy', verbose=1,
                         save_best_only=True, mode='max')
    history = model.fit(train_generator,
                        epochs=EPOCHS, callbacks=[es],
                        validation_data=val_generator)

    tf.saved_model.save(model, (make_sub_dir(category, 'saved_bp')))
    plot_acc_loss(history, name + '_' + category)



    base_model = model.get_layer('mobilenetv2_1.00_224')
    base_model.trainable = True

    # Fine tune from this layer onwards
    fine_tune_at = 100

    '''Freeze all the layers before the `fine_tune_at` layer'''
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    opt_learning_rate = 1e-5
    model.compile(loss=loss_function,
                  optimizer=tf.keras.optimizers.Adam(opt_learning_rate),
                  metrics=['accuracy'])

    model.summary()
    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    bm_ft = ModelCheckpoint(make_sub_dir(category, 'best_models'), monitor='val_accuracy',
                         verbose=1, save_best_only=True, mode='max')

    ft_es = EarlyStopping(monitor=FT_STOPPING_CONDITION, mode='min', verbose=1, patience=FT_STOPPING_PATIENCE)
    history_fine = model.fit(train_generator,
                             epochs=FT_EPOCHS,
                             callbacks=[ft_es, bm_ft],
                             validation_data=val_generator)
    train_time = time.time() - start_time
    plot_acc_loss(history_fine, 'ft_'+ FILE_NAME + '_' + category)

    '''Save optimized model in Keras format for re-training'''

    model.save(make_sub_dir(category, 'keras_models') + FILE_NAME + '.h5')

    convert_to_tf(FILE_NAME, model)

    layers_string = ', '.join([layer.name for layer in model.layers])

    tbl_path='tables/'+ category + '.csv'
    if os.path.isfile(tbl_path):
        table = pd.read_csv(tbl_path, index_col=0)
    else:
        if not os.path.exists('tables'):
            os.mkdir('tables')
        table=pd.DataFrame(columns=['name', 'time', 'opt_best_val_loss', 'opt_best_val_acc',
       'opt_last_val_loss', 'opt_last_val_acc', 'best_val_loss',
       'best_val_acc', 'last_val_loss', 'last_val_acc', 'sourcefile', 'loaded',
       'epochs', 'batch_size', 'stopping_cond', 'stop_p', 'optimizer', 'lr',
       'loss_f', 'opt_epochs', 'opt_batch_size', 'opt_stopping_cond',
       'opt_stop_p', 'opt_optimizer', 'opt_lr', 'opt_loss_f', 'layers',
       'training time'])

    table = table.append(pd.Series([FILE_NAME, train_time, min(history_fine.history['val_loss']),
                                    max(history_fine.history['val_accuracy']), history_fine.history['val_loss'][-1],
                                    history_fine.history['val_accuracy'][-1], min(history.history['val_loss']),
                                    max(history.history['val_accuracy']), history.history['val_loss'][-1],
                                    history.history['val_accuracy'][-1],
                                    image_dir, 'new_model', EPOCHS, BATCH_SIZE, STOPPING_CONDITION, STOPPING_PATIENCE,
                                    str(optimizer), 'default_lr', loss_function, FT_EPOCHS, BATCH_SIZE,
                                    FT_STOPPING_CONDITION, FT_STOPPING_PATIENCE, str(optimizer),
                                    str(tf.keras.backend.eval(model.optimizer.lr)), loss_function, layers_string,
                                    train_time], index=table.columns ), ignore_index=True)
    table.to_csv(tbl_path)
    return model


def train_multi(name, dataframe, image_dir, eepochs=12, ft_epochs=12, patience=5, ft_patience=4, old_dir=None):
    from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras import regularizers, optimizers
    import pandas as pd
    import numpy as np
    start_time=time.time()
    IMAGE_SIZE = 224
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    EPOCHS=eepochs
    FT_EPOCHS=ft_epochs
    STOPPING_CONDITION = 'val_loss'
    STOPPING_PATIENCE = patience
    FT_STOPPING_CONDITION = 'val_loss'
    FT_STOPPING_PATIENCE = ft_patience
    BATCH_SIZE=16
    loss_function = 'categorical_crossentropy'

    datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    train_generator = datagen.flow_from_dataframe(
        dataframe=dataframe[:2000],
        directory=image_dir,
        x_col="file name",
        y_col="combined",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(224, 224))
    valid_generator = test_datagen.flow_from_dataframe(
        dataframe=dataframe[2000:2700],
        directory=image_dir,
        x_col="file name",
        y_col="combined",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(224, 224))
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=dataframe[:1800],
        directory=image_dir,
        x_col="file name",
        y_col="combined",
        batch_size=1,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(224, 224))
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    inp = Input(shape=IMG_SHAPE)
    x = Conv2D(32, (3, 3), padding='same')(inp)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.20)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    output1 = Dense(1, activation='sigmoid')(x)
    output2 = Dense(1, activation='sigmoid')(x)
    model = Model(inp, [output1, output2])
    model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6),
                  loss=["binary_crossentropy", "binary_crossentropy"], metrics=["accuracy"])

    classes = train_generator.class_indices.keys()
    labels = '\n'.join(sorted(classes))
    number_of_categories = len(classes)
    with open('misc/multi_labels.txt', 'w') as f:
        f.write(labels)

    def generator_wrapper(generator):
        for batch_x, batch_y in generator:
            yield (batch_x, [batch_y[:, i] for i in range(2)])

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    model.fit_generator(generator_wrapper(valid_generator), steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data = generator_wrapper(valid_generator), validation_steps = STEP_SIZE_VALID,
                        epochs = 9, verbose = 2)

    test_generator.reset()
    pred = model.predict_generator(valid_generator, steps = STEP_SIZE_TEST, verbose = 1)
    tf.saved_model.save(model, 'multi_cat')

def compare_models(image_dir, test_sample, labels, prod_or_cat, model1, model2=None):
    products=[]
    categories=[]
    for index, row in test_sample.iterrows():
        products.append(row['product'])
        categories.append(row['category'])

    if prod_or_cat=='category':
        sample_labels=categories
    else:
        sample_labels=products


    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_sample,
        directory=image_dir,
        x_col="file name",
        y_col=prod_or_cat,
        batch_size=1,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(224, 224))

    predictions1 = model1.predict_generator(test_generator)
    pred_classes1 = []
    for pred in predictions1:
        pred_classes1.append(np.argmax(pred))
    right1 = 0
    for n in range(len(predictions1)):
        if sample_labels[n] == labels[pred_classes1[n]]:
            right1 += 1
    accuracy1 = right1 / len(products)

    if model2 is None:
        print('no old model found')
        accuracy2 = 100
    else:
        print('old model found')
        predictions2 = model2.predict_generator(test_generator)
        pred_classes2 = []
        for pred in predictions2:
            pred_classes2.append(np.argmax(pred))

        right2 = 0
        for n in range(len(predictions2)):
            if sample_labels[n] == labels[pred_classes2[n]]:
                right2 += 1

        accuracy2 = right2 / len(products)
        print(accuracy1)
        print(accuracy2)
    return(accuracy1, accuracy2)

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def check_labels_models(thing, models_dir, labels_dir):
    thing_model_dir = models_dir + '/' + thing
    thing_labels_path = labels_dir + '/' + thing + '/labels.txt'
    print(thing)
    if os.path.exists(thing_model_dir):
        print('Old ' + thing + " model exists at " + thing_model_dir)
        model_exists = True
    else:
        print('No old ' + thing + " model exists at " + thing_model_dir)
        model_exists = False
    try:
        with open(thing_labels_path, 'r') as f:
            unique_labels = (f.read()).split('\n')
            unique_labels = list(filter(None, unique_labels))

            labels_loaded = True
        print('Old labels exist for ' + thing + ', at ' + thing_labels_path)
    except:
        print('Old labels don\'t exist for ' + thing + ', at ' + thing_labels_path)
        unique_labels = []
        labels_loaded = False

    return model_exists and labels_loaded, unique_labels
