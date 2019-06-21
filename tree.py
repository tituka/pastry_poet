import balance
from PIL import Image
import shutil
import os
from datetime import date
import pickle
today = date.today()

d1 = today.strftime("%d-%m-%Y")
root_new_data='try_balance'
directories=[d for d in os.listdir(root_new_data)]
median=balance.get_median_size(root_new_data)
'''checks that no new categories have been added'''
if directories==pickle.load(open('trained_categories.pickle','rb')):
	'''checks that there is enough new data to traim'''
	if median>=7:
		for dr in directories:
			print "in directory" + dr
			pics=[]
			for picfile in os.listdir(root_new_data + '/' + dr + '/'):
				pics.append(Image.open(root_new_data+'/'+dr+'/'+picfile))
			balance.balance(dr, pics, median , 'destination/')
		os.mkdir('balanced-'+ d1)

		for f in directories:
		        shutil.move(root_new_data+'/' +f, 'balanced-'+ d1)
	else:
		'not enough new pictures'
