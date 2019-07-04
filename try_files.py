import pandas as pd
import tools
import numpy as np
import PIL.Image as Image
import tensorflow as tf

IMAGE_SIZE = 224
IMG_SHAPE = [IMAGE_SIZE, IMAGE_SIZE]

image_dir='/home/tiina/vegetables/all_in_one/'
#table= pd.read_csv('t.csv', dtype=str)
#dropped=pd.read_csv('dropped_table.csv', dtype=str)


'''test_sample_num=int(len(table)*.05)
test_sample=table.sample(test_sample_num)
table=table.drop(test_sample.index.values)'''
images=[]
products=[]
categories=[]

#test_sample=table[~table['file name'].isin(dropped['file name'])]
model1 = tf.keras.models.load_model('best_models/all/OVERTRAIN_2')
model2 = tf.keras.models.load_model('keras_models/all/OVERTRAIN_2.h5')
test_sample=pd.read_csv('sample.csv')
for index, row in test_sample.iterrows():
    products.append(row['product'])
    categories.append(row['category'])
    file_path=image_dir+'/'+row['file name']
    image_opened=Image.open(file_path).resize(IMG_SHAPE)
    images.append(np.array(image_opened)/255.0)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_sample,
    directory=image_dir,
    x_col="file name",
    y_col="category",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(224, 224))



predictions1=model1.predict_generator(test_generator)
predictions2=model2.predict_generator(test_generator)

pred_classes1=[]
for pred in predictions1:
    pred_classes1.append(np.argmax(pred))

pred_classes2=[]
for pred in predictions2:
    pred_classes2.append(np.argmax(pred))

with open('misc/all/labels.txt', 'r') as f:
   class_names=(f.read()).split('\n')

print(class_names)
right1 = 0
for n in range(len(predictions1)):
    if categories[n] == class_names[pred_classes1[n]]:
        right1 += 1

right2=0
for n in range(len(predictions2)):
    if categories[n] == class_names[pred_classes2[n]]:
        right2 += 1


accuracy1=right1/len(products)
accuracy2=right2/len(products)

print(accuracy1)
print(accuracy2)



