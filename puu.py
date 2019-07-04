import pandas as pd
import tools
import tensorflow as tf
import numpy as np
import tensorflow as tf
import os
from tools import compare_models
import time
import sys


def train_cycle():
    base_dir = '/home/tiina'
    start_time = time.time()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    image_dir = base_dir + '/vegetables/all_in_one/'
    table_orig = pd.read_csv('t.csv', dtype=str)
    table_shuffled = table_orig.sample(frac=1)
    old_table = pd.read_csv('t.csv', dtype=str)

    test_sample_num = int(len(table_orig) * .2)
    test_sample = table_shuffled.sample(test_sample_num)
    table = table_shuffled.drop(test_sample.index.values)
    test_sample.to_csv('sample_shuffled.csv')
    table.to_csv('table_shuffled.csv')

    # test_sample=pd.read_csv('sample_shuffled.csv')
    # table=pd.read_csv('table_shuffled.csv')

    test_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
    models_dir=base_dir + '/poet/pastry_poet/used_models'
    labels_dir=base_dir+ '/poet/pastry_poet/misc'
    acc_dict=dict()

    try_retrain, unique_old_labels= tools.check_labels_models('all', models_dir, labels_dir)
    if try_retrain:
        print('Trying to retrain all')
        old_model_dir = models_dir + '/all'
        print(table_orig.category.unique())
        if set(table_orig.category.unique())==set(unique_old_labels):
            try:
                print('Matching model to retrain exists for all')
                old_model = tf.keras.models.load_model(old_model_dir)
                print('Old model for all successfully loaded from ' + old_model_dir)
                new_model=tools.train(name='all', dataframe=table, image_dir=image_dir, category='all', epochs=10, ft_epochs=10, patience=2, ft_patience=0, train_categories=True,
                                      old_dir=models_dir + '/all')
                print('New model for all successfully trained on top of ' + old_model_dir)

                acc_dict.update({'all': compare_models(image_dir, test_sample, unique_old_labels, 'category',  model1=new_model, model2=old_model)})

            except:
                print('Failure loading previous model despite matching label file from ' +  old_model_dir)
                new_model = tools.train(name='all', dataframe=table, image_dir=image_dir, category='all', epochs=10,
                                        ft_epochs=10, patience=0, ft_patience=0, train_categories=True)
                acc_dict.update({'all': compare_models(image_dir, test_sample, unique_old_labels, 'category', new_model)})
        else:
            print('No matching labels to retrain for all')
            print(image_dir)

            new_model = tools.train(name='all', dataframe=table, image_dir=image_dir, category='all', epochs=10,
                                    ft_epochs=10, patience=0, ft_patience=0, train_categories=True)
            acc_dict.update({'all': compare_models(image_dir, test_sample, unique_old_labels, 'category', new_model)})

    else:
        print('No matching model to retrain for all')
        new_model = tools.train(name='all', dataframe=table, image_dir=image_dir, category='all', epochs=10, ft_epochs=10, patience=0, ft_patience=0, train_categories=True)
        acc_dict.update({'all':compare_models(image_dir, test_sample, unique_old_labels, 'category',  new_model)})

    for cat in table_orig.category.unique():
        try_retrain, unique_old_labels = tools.check_labels_models(cat, models_dir, labels_dir)
        new_products=table_orig.loc[table_orig['category'] == cat]['product'].unique()
        old_products=old_table.loc[old_table['category'] == cat]['product'].unique()
        cat_prod_table = table.loc[table['category'] == cat]
        cat_test_table=test_sample.loc[test_sample['category'] == cat]
        print('Test table rows for ' + cat  +': '+str(len(cat_test_table)))
        if try_retrain:
            print('Trying to retrain ' + cat)
            if set(new_products)==set(unique_old_labels):
                old_model_dir = models_dir + '/' + cat
                try:
                    print('Matching model to retrain exists for ' + cat)
                    old_model = tf.keras.models.load_model(old_model_dir)
                    print("Old model for " +cat+ " successfully loaded from " + old_model_dir)
                    new_model = tools.train(name=cat, dataframe=cat_prod_table, image_dir=image_dir, category=cat, epochs=10, ft_epochs=10, patience=0, ft_patience=0,
                                            old_dir=models_dir+'/' +cat)
                    print('New model for '+ cat +' successfully trained on top of ' + old_model_dir)

                    acc_dict.update({cat: compare_models(image_dir, cat_test_table, unique_old_labels, 'product',  model1=new_model, model2=old_model)})
                except:
                    print('Failure loading previous model despite matching label file from '+ old_model_dir)
                    new_model = tools.train(name=cat, dataframe=cat_prod_table, image_dir=image_dir, category=cat, epochs=10,
                                            ft_epochs=10, patience=0, ft_patience=0)
                    acc_dict.update({cat: compare_models(image_dir, cat_test_table, unique_old_labels, 'product', new_model)})
            else:
                print('No matching labels to retrain for ' + cat)
                new_model = tools.train(name=cat, dataframe=cat_prod_table, image_dir=image_dir, category=cat, epochs=10,
                                        ft_epochs=10, patience=0, ft_patience=0)
                acc_dict.update({cat: compare_models(image_dir, cat_test_table, unique_old_labels, 'product', new_model)})


        else:
            print('No matching model exists to retrain for ' + cat)
            new_model = tools.train(name=cat, dataframe=cat_prod_table, image_dir=image_dir, category=cat,epochs=10, ft_epochs=10, patience=0, ft_patience=0)
            acc_dict.update({cat: compare_models(image_dir, cat_test_table, unique_old_labels, 'product', new_model)})

    for m in acc_dict.keys():
        print(m)
        print(acc_dict[m][0])
        print(acc_dict[m][1])
        if acc_dict[m][0] < acc_dict[m][1]:
            if acc_dict[m][1] == 100:
                print('New labels for ' + m + ', whole new model with accuracy ' + str(acc_dict[m][0]))
                tools.copy_and_overwrite('/home/tiina/poet/pastry_poet/best_models/' + m,
                                         '/home/tiina/poet/pastry_poet/used_models/' + m)
                print('Saved to ' + models_dir +'/' + m)
            else:
                print('Retrained model accuracy ' + str(acc_dict[m][0]) + ', lower than old ' + str(acc_dict[m][1])+ '/n Not saved. ')


        else:
            print('Retrained model accuracy ' + str(acc_dict[m][0]) + ', higher than old ' + str(acc_dict[m][1]))
            tools.copy_and_overwrite('/home/tiina/poet/pastry_poet/best_models/' + m,
                                     '/home/tiina/poet/pastry_poet/used_models/' + m)
            print('Saved to ' + models_dir + '/' + m)

    print('Took time: '+ str(time.time() - start_time))


'''
def double_smallest(df, image_dir, root_dir, minumun_size):
    p_lens=[]
    for cat in df.category.unique():
        for prod in df.loc[test_sample['category'] == cat]:
            p_lens.append(len(df.loc[test_sample['category'] == cat]))
    if min(p_lens)



'''
