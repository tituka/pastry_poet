import os
from PIL import Image
import random
import numpy as np

def balance(category_name, category_pics, new_total, destination):
	if len(category_pics)<=new_total:
			balanced_category=add_mirrored(category_pics, new_total)
	else:
			balanced_category=category_pics[:int(new_total)]

	os.mkdir('destination/'+ category_name)	
	n=0
	for pic in balanced_category:
		pic.save('destination/'+ category_name+'/' + str(n) + '.png')
		n+=1



def add_mirrored(pics, how_many):
	mirrored=[]
	final=pics
	for pic in pics:
		if len(final) + len(mirrored)< how_many:
			print 'first mirroring'
			mirrored_pic=pic.transpose(Image.FLIP_LEFT_RIGHT)
			mirrored.append(mirrored_pic)
			final.append(mirrored_pic)
	odd=True
	print str(len(final))
	while len(final)<how_many-len(mirrored):
		print len(final)
		print len(mirrored)
		print 'whole lot less'
		if odd:
			print 'odd'
			final=final+mirrored
			odd= False
		else:
			print 'even'
			final=final+pics
			odd=True			
	if len(final)< how_many:
		print 'leftovers'
		print str(how_many-len(final))
		if odd:
			final=final+mirrored[:int(how_many-len(final))]
		else:
			final= final+final[:int(how_many-len(final))]

	return final

def get_median_size(root_dir):
	sizes=[len(os.listdir(root_dir+'/'+ x)) for x in os.listdir(root_dir)]
	return np.median(sizes)
