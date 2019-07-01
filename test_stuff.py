import tensorflow as tf
import numpy as np
import PIL.Image as Image

def predict_image(model, labels_path, image_location):
    FILENAME=image_location
    labels = np.array(open(labels_path).read().splitlines())
    IMAGE_SIZE = 224
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    pic = Image.open().resize(IMG_SHAPE)
    pic_processed = np.array(pic)/255.0
    result = model.predict(pic_processed[np.newaxis, ...])
    result.shape
    predicted_class = np.argmax(result[0], axis=-1)
    return  predicted_class
