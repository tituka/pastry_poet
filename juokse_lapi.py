import pandas as pd
import tools
import numpy as np
import PIL.Image as Image
import tensorflow as tf

IMAGE_SIZE = 224
IMG_SHAPE = [IMAGE_SIZE, IMAGE_SIZE]

image_dir='/home/tiina/vegetables/all_in_one/'
table= pd.read_csv('t.csv', dtype=str)


test_sample_num=int(len(table)*.20)
test_sample=table.sample(test_sample_num)
table=table.drop(test_sample.index.values)
test_sample.to_csv('sample.csv')

images=[]
products=[]
categories=[]


for index, row in test_sample.iterrows():
    products.append(row['product'])
    categories.append(row['category'])
    file_path=image_dir+'/'+row['file name']
    image_opened=Image.open(file_path).resize(IMG_SHAPE)
    images.append(np.array(image_opened)/255.0)

tools.train(dataframe=table, image_dir=image_dir, name='OVERTRAIN_2', category='all', epochs=50, ft_epochs=50, patience=30, ft_patience=30,
            train_categories=True)

