import logging
console = logging.FileHandler('logs/try_own', 'a')
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('puu').addHandler(console)

import pandas as pd
from tools import compare_models
import tools
import tensorflow as tf
import numpy as np
import tensorflow as tf
import os
import time
import sys
import logging

def train_all():
    EPOCHS = 20
    FT_EPOCHS = 20
    PATIENCE = 3
    FT_PATIENCE = 3
    base_dir = '/home/tiina'
    start_time = time.time()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    image_dir = base_dir + '/vegetables/all_in_one/'
    table_orig = pd.read_csv('t.csv', dtype=str)
    table_shuffled = table_orig.sample(frac=1)

    test_sample_num = int(len(table_orig) * .18)
    test_sample = table_shuffled.sample(test_sample_num)
    table = table_shuffled.drop(test_sample.index.values)
    test_sample.to_csv('sample_shuffled_all.csv')
    table.to_csv('table_shuffled_all.csv')

    models_dir = base_dir + '/poet/pastry_poet/used_models'
    labels_dir = base_dir + '/poet/pastry_poet/misc'
    new_model = tools.train(name='every_product', dataframe=table, image_dir=image_dir, category='product', epochs=EPOCHS,
                            ft_epochs=FT_EPOCHS, patience=PATIENCE, ft_patience=FT_PATIENCE,
                            train_categories=False)



def train_cycle():
    '''
    # get TF logger
    log_tf = logging.getLogger('tensorflow')
    log_tf.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    fh = logging.FileHandler('tensorflow.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log_tf.addHandler(fh)
    '''


    EPOCHS = 1
    FT_EPOCHS = 1
    PATIENCE = 0
    FT_PATIENCE = 0
    base_dir = '/home/tiina'
    start_time = time.time()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    image_dir = base_dir + '/vegetables/all_in_one/'
    table_orig= pd.read_csv('t.csv', dtype=str)
    table_shuffled = table_orig.sample(frac=1)
    logging.error('This should go to both console and file')
    test_sample_num = int(len(table_orig) * .2)
    test_sample = table_shuffled.sample(test_sample_num)
    table = table_shuffled.drop(test_sample.index.values)
    test_sample.to_csv('sample_shuffled.csv')
    table.to_csv('table_shuffled.csv')

    # test_sample=pd.read_csv('sample_shuffled.csv')
    # table=pd.read_csv('table_shuffled.csv')

    models_dir = base_dir + '/poet/pastry_poet/used_models'
    labels_dir = base_dir + '/poet/pastry_poet/misc'
    acc_dict = dict()

    try_retrain, unique_old_labels= tools.check_labels_models('all', models_dir, labels_dir)
    if try_retrain:
        logging.info('Trying to retrain all')
        old_model_dir = models_dir + '/all'
        logging.info(table_orig.category.unique())
        if set(table_orig.category.unique()) == set(unique_old_labels):
            try:
                logging.info('Matching model to retrain exists for all')
                old_model = tf.keras.models.load_model(old_model_dir)
                logging.info('Old model for all successfully loaded from ' + old_model_dir)
                new_model = tools.train(name='all', dataframe=table, image_dir=image_dir, category='all', epochs=EPOCHS,
                                        ft_epochs=FT_EPOCHS, patience=PATIENCE, ft_patience=FT_PATIENCE,
                                        train_categories=True, old_dir=models_dir + '/all')
                logging.info('New model for all successfully trained on top of ' + old_model_dir)

                acc_dict.update({'all': compare_models(image_dir, test_sample, unique_old_labels, 'category',
                                                       model1=new_model, model2=old_model)})
                del old_model
                del new_model
            except:
                logging.info('Failure loading previous model despite matching label file from ' + old_model_dir)
                new_model = tools.train(name='all', dataframe=table, image_dir=image_dir, category='all', epochs=EPOCHS,
                                        ft_epochs=FT_EPOCHS, patience=PATIENCE, ft_patience=FT_PATIENCE,
                                        train_categories=True)
                acc_dict.update({'all': compare_models(image_dir, test_sample, unique_old_labels, 'category',
                                                       new_model)})
                del new_model
        else:
            logging.info('No matching labels to retrain for all')
            logging.info(image_dir)
            new_model = tools.train(name='all', dataframe=table, image_dir=image_dir, category='all', epochs=EPOCHS,
                                    ft_epochs=FT_EPOCHS, patience=PATIENCE, ft_patience=FT_PATIENCE,
                                    train_categories=True)
            acc_dict.update({'all': compare_models(image_dir, test_sample, unique_old_labels, 'category', new_model)})
            del new_model
    else:
        logging.info('No matching model to retrain for all')
        new_model = tools.train(name='all', dataframe=table, image_dir=image_dir, category='all', epochs=EPOCHS,
                                ft_epochs=FT_EPOCHS, patience=PATIENCE, ft_patience=FT_PATIENCE,
                                train_categories=True)
        acc_dict.update({'all': compare_models(image_dir, test_sample, unique_old_labels, 'category',  new_model)})
        del new_model
    for cat in table_orig.category.unique():
        try_retrain, unique_old_labels = tools.check_labels_models(cat, models_dir, labels_dir)
        new_products = table_orig.loc[table_orig['category'] == cat]['product'].unique()
        cat_prod_table = table.loc[table['category'] == cat]
        cat_test_table = test_sample.loc[test_sample['category'] == cat]
        logging.info('Test table rows for ' + cat + ': ' + str(len(cat_test_table)))
        if try_retrain:
            logging.info('Trying to retrain ' + cat)
            if set(new_products) == set(unique_old_labels):
                old_model_dir = models_dir + '/' + cat
                try:
                    logging.info('Matching model to retrain exists for ' + cat)
                    old_model = tf.keras.models.load_model(old_model_dir)
                    logging.info("Old model for " + cat + " successfully loaded from " + old_model_dir)
                    new_model = tools.train(name=cat, dataframe=cat_prod_table, image_dir=image_dir, category=cat,
                                            epochs=EPOCHS, ft_epochs=FT_EPOCHS, patience=PATIENCE,
                                            ft_patience=FT_PATIENCE, old_dir=models_dir+'/' + cat)
                    logging.info('New model for ' + cat + ' successfully trained on top of ' + old_model_dir)
                    acc_dict.update({cat: compare_models(image_dir, cat_test_table, unique_old_labels, 'product',
                                                         model1=new_model, model2=old_model)})
                    del old_model
                    del new_model
                except:
                    logging.info('Failure loading previous model despite matching label file from '+ old_model_dir)
                    new_model = tools.train(name=cat, dataframe=cat_prod_table, image_dir=image_dir, category=cat,
                                            epochs=EPOCHS, ft_epochs=FT_EPOCHS, patience=PATIENCE,
                                            ft_patience=FT_PATIENCE)
                    acc_dict.update({cat: compare_models(image_dir, cat_test_table, unique_old_labels, 'product',
                                                         new_model)})
                    del new_model
            else:
                logging.info('No matching labels to retrain for ' + cat)
                new_model = tools.train(name=cat, dataframe=cat_prod_table, image_dir=image_dir, category=cat,
                                        epochs=EPOCHS, ft_epochs=FT_EPOCHS, patience=PATIENCE, ft_patience=FT_PATIENCE)
                acc_dict.update({cat: compare_models(image_dir, cat_test_table, unique_old_labels, 'product', new_model)})
                del new_model

        else:
            logging.info('No matching model exists to retrain for ' + cat)
            new_model = tools.train(name=cat, dataframe=cat_prod_table, image_dir=image_dir, category=cat,eepochs=EPOCHS,
                                    ft_epochs=FT_EPOCHS, patience=PATIENCE, ft_patience=FT_PATIENCE)
            acc_dict.update({cat: compare_models(image_dir, cat_test_table, unique_old_labels, 'product', new_model)})
            del new_model
        del cat_test_table
        del cat_prod_table
    del table
    del table_orig
    del table_shuffled
    del test_sample

    for m in acc_dict.keys():
        logging.info(m)
        logging.info(acc_dict[m][0])
        logging.info(acc_dict[m][1])
        if acc_dict[m][0] < acc_dict[m][1]:
            if acc_dict[m][1] == 100:
                logging.info('New labels for ' + m + ', whole new model with accuracy ' + str(acc_dict[m][0]))
                tools.copy_and_overwrite('/home/tiina/poet/pastry_poet/best_models/' + m,
                                         '/home/tiina/poet/pastry_poet/used_models/' + m)
                logging.info('Saved to ' + models_dir + '/' + m)
                model_to_convert = tf.keras.models.load_model('/home/tiina/poet/pastry_poet/best_models/' +  m)
                tools.convert_to_tf(m, model_to_convert)
                del model_to_convert
                logging.info('Converted to tflite model')
            else:
                logging.info('Retrained model accuracy for ' +m+ ' is '+ str(acc_dict[m][0]) + ', lower than old ' + str(acc_dict[m][1]))
                logging.info('Model not saved or converted to tflite.')


        else:
            logging.info('Retrained model accuracy for ' + m + ' is ' +  str(acc_dict[m][0]) + ', higher than old ' + str(acc_dict[m][1]))
            tools.copy_and_overwrite('/home/tiina/poet/pastry_poet/best_models/' + m,
                                     '/home/tiina/poet/pastry_poet/used_models/' + m)
            logging.info('Saved to ' + models_dir + '/' + m)
            model_to_convert = tf.keras.models.load_model('/home/tiina/poet/pastry_poet/best_models/' + m)
            tools.convert_to_tf(m, model_to_convert)
            del model_to_convert
            logging.info('Converted to tflite model')

    logging.info('Took time: ' + str(time.time() - start_time))




if __name__ == '__main__':

    train_cycle()