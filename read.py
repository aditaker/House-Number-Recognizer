import scipy.io as sio
import numpy as np
import pandas as pd
import hdf5storage
import h5py
from PIL import Image

dir_path = "Dataset/"


def pre_process(foldername):
	""" Reads and processes the mat files provided in the SVHN dataset. 
		Input: filename 
		Ouptut: list of python dictionaries 
	""" 
		
	f = h5py.File(dir_path+foldername+'/digitStruct.mat', 'r')
	groups = f['digitStruct'].items()
	bbox_ds = np.array(groups[0][1]).squeeze()
	names_ds = np.array(groups[1][1]).squeeze()

	num_files = bbox_ds.shape[0]
	count = 0
	inp = []
	out = []
	for objref1, objref2 in zip(bbox_ds, names_ds):


		# Extract image name
		names_ds = np.array(f[objref2]).squeeze()
		filename = ''.join(chr(x) for x in names_ds)
		img = Image.open(dir_path+foldername+'/'+filename)
		#print filename

		# Extract other properties
		items1 = f[objref1].items()

		# Extract image label
		labels_ds = np.array(items1[1][1]).squeeze()
		try:
			label_vals = [int(f[ref][:][0, 0]) for ref in labels_ds]
		except TypeError:
			label_vals = [labels_ds]
		length_labels = len(label_vals)

		if( length_labels > 5):
			continue

		temp = np.full((1,5),10)
		if(length_labels==1):
			label_vals[0] = int(label_vals[0]);
		labels = 0
		print label_vals
		for x in label_vals:
			if x == 10:
				x=0;
			temp[0][labels] = x;
			labels += 1;
			#=print labels

		# Extract image height
		height_ds = np.array(items1[0][1]).squeeze()
		try:
			height_vals = [f[ref][:][0, 0] for ref in height_ds]
		except TypeError:
			height_vals = [height_ds]

		# Extract image left coords
		left_ds = np.array(items1[2][1]).squeeze()
		try:
			left_vals = [f[ref][:][0, 0] for ref in left_ds]
		except TypeError:
			left_vals = [left_ds]

		# Extract image top coords
		top_ds = np.array(items1[3][1]).squeeze()
		try:
			top_vals = [f[ref][:][0, 0] for ref in top_ds]
		except TypeError:
			top_vals = [top_ds]

		# Extract image width
		width_ds = np.array(items1[4][1]).squeeze()
		try:
			width_vals = [f[ref][:][0, 0] for ref in width_ds]
		except TypeError:
			width_vals = [width_ds]

		x_bottom = [(x + y) for x, y in zip(left_vals, width_vals)]
		y_bottom = [(x + y) for x, y in zip(top_vals, height_vals)]


		x0 = int(min(left_vals)*0.85)
		y0 = int(min(top_vals)*0.85)
		x1 = min(int(max(x_bottom)*1.15),img.size[0])
		y1 = min(int(max(y_bottom)*1.15),img.size[1])
		img = img.crop((x0,y0,x1,y1))
		img = img.resize((32, 32), Image.ANTIALIAS)
		a = np.asarray(img);
		#img.save(foldername + '_processed/' + filename)
		inp.append(a)
		out.append(temp)

		count += 1
		print 'Processed: %d/%d' % (count, num_files)



	return np.asarray(inp, dtype = np.float32),np.asarray(np.reshape(out,(len(out),5)), dtype = np.int64)

	#df = pd.DataFrame(data_list);
	#df.to_csv('foo.csv')
	#return data_list
#read_boundingbox_data(dir_path+'train/digitStruct.mat');
#print(train_bounding_box);
#x = read_process_h5(dir_path+'train/digitStruct.mat')
#print x[0]


# def create_bounding_boxes(data_list):
# 	data_list = read_process_h5(dir_path+'train/digitStruct.mat')
# 	bboxes = []
# 	for data in data_list:
# 		for i in range(data['length']):
# 			bboxes.append({
# 				'filename': data['filename'],
# 				'label': data['labels'][i],
# 				'x_top': data['left'][i],
# 				'y_top': data['top'][i],
# 				'x_bottom': data['left'][i]+data['width'][i],
# 				'y_bottom': data['top'][i]+data['height'][i]
# 				});
# 	print bboxes[5]['filename']
# 	df = pd.DataFrame(bboxes);
# 	df.to_csv('foo-bar.csv')
# 	image = Image.open(dir_path+'train/'+bboxes[5]['filename']);
# 	# Use draw module can be used to annotate the image
# 	draw = ImageDraw.Draw(image)
		
# 	# Bounding box rectangle [x0, y0, x1, y1]
# 	rectangle = [bboxes[5]['x_top'], bboxes[5]['y_top'], bboxes[5]['x_bottom'], bboxes[5]['x_bottom']]    
# 	# Draw a rectangle on top of the image
# 	draw.rectangle(rectangle, outline="red")


# #create_bounding_boxes([]);
# read_process_h5(dir_path+'train/digitStruct.mat')
