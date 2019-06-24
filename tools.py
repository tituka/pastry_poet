import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import shutil
import tensorflow as tf


def plot_acc_loss(mhistory, mname):
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


def convert_to_tf(save_file_name, model, delete_pb_model=True, save_converted=True):
    saved_model_dir = 'pb_for_tf_models/' + save_file_name
    tf.saved_model.save(model, saved_model_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    if save_converted:
        with open('tflite_models/'+ save_file_name+'.tflite', 'wb') as f:
          f.write(tflite_model)
    if delete_pb_model:
        shutil.rmtree('pb_for_tf_models/' + save_file_name)
    return tflite_model