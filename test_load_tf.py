import tensorflow as tf
import pandas as pd
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model_own2.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_shape = input_details[0]['shape']
input_details
input_shape
data_root = "/home/tiina/vegetables/webcam"

IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = str(data_root)

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(TRAINING_DATA_DIR,subset="validation",shuffle=False,target_size=IMAGE_SHAPE)
val_image_batch, val_label_batch = next(iter(valid_generator))

interpreter.resize_tensor_input(input_details[0]['index'], (32, 224, 224, 3))
interpreter.resize_tensor_input(output_details[0]['index'], (32, 5))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
# Set batch of images into input tensor
interpreter.set_tensor(input_details[0]['index'], val_image_batch)
# Run inference
interpreter.invoke()
# Get prediction results
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", tflite_model_predictions.shape)

# Convert prediction results to Pandas dataframe, for better visualization
tflite_pred_dataframe = pd.DataFrame(tflite_model_predictions)
dataset_labels = [line.rstrip('\n') for line in open('labels_own.txt')]
tflite_pred_dataframe.columns = dataset_labels
interpreter.invoke()
# Get prediction results
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", tflite_model_predictions.shape)


# Get predictions for each image
predicted_ids = np.argmax(tflite_model_predictions, axis=-1)
predicted_labels = np.array(dataset_labels)[predicted_ids.astype(int)]

import matplotlib.pyplot as plt

# Print images batch and labels predictions
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
true_label_ids=np.argmax(val_label_batch, axis=-1)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(val_image_batch[n])
  color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.savefig('jee.png')
