import os
import json
import math
import numpy as np 
import matplotlib.image as mpimg
import tensorflow as tf
from skimage.transform import resize
import random

class Data_processor:
	# data_for: 'train','val'
	# for posetrack18 and pose estimation training
	def __init__(self,data_dir,anno_dir,model_input_shape,model_output_shape,batch_size_select,data_for):
		self.bbox = []
		self.kps_and_valid = []
		self.kps = []
		self.scaled_kps = []
		self.valids = []
		self.track_ids = []
		self.img_ids = []
		self.id_to_file_dict = {}
		self.start_idx = []
		self.end_idx = []
		self.input_shape = model_input_shape
		self.output_shape = model_output_shape
		self.n_imgs = None
		self.batch_size = batch_size_select
		self.n_batchs = None
		self.data_dir = data_dir
		self.anno_dir = anno_dir
		self.data_for = data_for

		self.get_data_from_dir()
		self.get_start_end_idx()
		self.get_target_valid_joint()
		self.scale_bbox_to_43()
		self.convert_kps_to_local()
		self.scale_kps_to_output_size()





	def get_data_from_dir(self):
		temp_image_id_with_label = []
		temp_file_name_with_label = []
		temp_anno_bbox_list = []
		temp_anno_track_id_list = []
		temp_anno_id_list = []
		temp_anno_kp_list = []

		temp_anno_dir = self.anno_dir + self.data_for + '/'
		for anno_file in os.listdir(temp_anno_dir):
			if anno_file.endswith('.json'):
				temp = temp_anno_dir + anno_file
				with open(temp) as f:
					data = json.load(f)
				data_images = data['images']
				data_annotations = data['annotations']
			for temp_image in data_images:
				if temp_image['is_labeled']:
					temp_image_id_with_label.append(temp_image['id'])
					temp_file_name_with_label.append(temp_image['file_name'])
			
			for anno in data_annotations:
				temp_keys = list(anno.keys())
				to_check_keys = ['image_id','bbox','track_id','image_id','keypoints']
				if all(item in temp_keys for item in to_check_keys):
					if anno['image_id'] in temp_image_id_with_label:
						bbox_temp = anno['bbox']
						if bbox_temp[2] > 210 and bbox_temp[3] > 280:
							if bbox_temp[0] >= 0 and bbox_temp[1] >= 0:
								temp_anno_bbox_list.append(anno['bbox'])
								temp_anno_track_id_list.append(anno['track_id'])
								temp_anno_id_list.append(anno['image_id'])
								temp_anno_kp_list.append(anno['keypoints'])
		temp_id_to_file_dict = {temp_image_id_with_label[i]:temp_file_name_with_label[i] for i in range(len(temp_image_id_with_label))}

		self.n_imgs = len(temp_anno_bbox_list)
		self.bbox = temp_anno_bbox_list
		self.kps_and_valid = temp_anno_kp_list
		self.track_ids = temp_anno_track_id_list
		self.img_ids = temp_anno_id_list
		self.id_to_file_dict = temp_id_to_file_dict


	def get_start_end_idx(self):
		max_idx = self.n_imgs
		temp_batch_size = self.batch_size
		l = list(range(max_idx))
		temp_start_idx = l[0::temp_batch_size]
		def add_batch_size(num,max_id=max_idx,bz=temp_batch_size):
			return min(num+bz,max_id)
		temp_end_idx = list(map(add_batch_size,temp_start_idx))
		self.start_idx = temp_start_idx
		self.end_idx = temp_end_idx
		self.n_batchs = len(temp_start_idx)

	def get_target_valid_joint(self):
		temp_anno_kp_valid = self.kps_and_valid
		temp_kps = []
		temp_valids = []
		for temp_anno_kp in temp_anno_kp_valid:
			temp_x = np.array(temp_anno_kp[0::3])
			temp_y = np.array(temp_anno_kp[1::3])
			temp_valid = np.array(temp_anno_kp[2::3])
			temp_valid = temp_valid.astype('float32')
			temp_target_coord = np.stack([temp_x,temp_y],axis = 1)
			temp_target_coord = temp_target_coord.astype('float32')

			temp_kps.append(temp_target_coord)
			temp_valids.append(temp_valid)
		self.kps = temp_kps
		self.valids = temp_valids

	def scale_bbox_to_43(self):
		temp_bbox = self.bbox
		temp_img_ids = self.img_ids
		temp_dict = self.id_to_file_dict
		for i in range(len(temp_img_ids)):
			i_img_id = temp_img_ids[i]
			i_dir_temp = temp_dict[i_img_id]
			i_dir_temp = self.data_dir + i_dir_temp

			i_img = mpimg.imread(i_dir_temp)
			i_shape = i_img.shape
			img_h = i_shape[0]
			img_w = i_shape[1]

			i_bbox = temp_bbox[i]
			bbox_x, bbox_y, bbox_w, bbox_h = i_bbox

			to_check = 0.75*bbox_h
			if to_check >= bbox_w:
				add_x = True
			else:
				add_x = False

			if add_x:
				new_bbox_h = bbox_h
				new_bbox_w = 0.75*bbox_h
				diff = new_bbox_w - bbox_w
				new_bbox_y = bbox_y
				new_bbox_x = bbox_x - 0.5*diff
				#check if in image
				if new_bbox_x < 0:
					new_bbox_x = 0
				if new_bbox_x+new_bbox_w >= img_w:
					new_bbox_x = img_w - new_bbox_w - 1
			else:
				new_bbox_w = bbox_w
				new_bbox_h = 4.0/3.0 * bbox_w
				diff = new_bbox_h - bbox_h
				new_bbox_x = bbox_x
				new_bbox_y = bbox_y - 0.5 * diff
				if new_bbox_y < 0:
					new_bbox_y = 0
				if new_bbox_y+new_bbox_h >= img_h:
					new_bbox_y = img_h - new_bbox_h - 1

			temp_new_bbox = [new_bbox_x,new_bbox_y,new_bbox_w,new_bbox_h]

			temp_bbox[i] = temp_new_bbox

		self.bbox = temp_bbox

	def convert_kps_to_local(self):
		temp_bbox = self.bbox
		temp_kps = self.kps
		for i in range(len(temp_kps)):
			i_bbox = temp_bbox[i]
			i_kps = temp_kps[i]
			i_origin = np.array([i_bbox[0], i_bbox[1]])
			for j in range(len(i_kps)):
				if i_kps[j,0] != 0:
					i_kps[j,:] = i_kps[j,:] - i_origin
			temp_kps[i] = i_kps
		self.kps = temp_kps

	def scale_kps_to_output_size(self):
		temp_bbox = self.bbox
		temp_kps = self.kps
		temp_output_shape = self.output_shape
		for i in range(len(temp_bbox)):
			i_bbox = temp_bbox[i]
			i_kps = temp_kps[i]
			scale_x = temp_output_shape[1] / i_bbox[2]
			scale_y = temp_output_shape[0] / i_bbox[3]
			temp_scale = np.array([scale_x, scale_y])
			for j in range(len(i_kps)):
				temp_value = i_kps[j,:]
				i_kps[j,:] = np.multiply(temp_value,temp_scale)
			temp_kps[i] = i_kps
		self.scaled_kps = temp_kps

	def render_gaussian_heatmap(self,input_kps,sigma):
		r_output_shape = self.output_shape
		x = [i for i in range(r_output_shape[1])]
		y = [i for i in range(r_output_shape[0])]

		xx,yy = tf.meshgrid(x,y)
		xx = tf.reshape(tf.cast(xx,tf.float32),(1,*r_output_shape,1))
		yy = tf.reshape(tf.cast(yy,tf.float32),(1,*r_output_shape,1))

		input_kps_float = input_kps.astype(np.float64)

		x = tf.floor(tf.reshape(input_kps_float[:,0],[-1,1,1,17])+ 0.5 )
		y = tf.floor(tf.reshape(input_kps_float[:,1],[-1,1,1,17])+ 0.5 )
		x = tf.cast(x,tf.float32)
		y = tf.cast(y,tf.float32)
		temp_heatmap = tf.exp(-(((xx-x)/tf.cast(sigma,tf.float32))**2)/tf.cast(2,tf.float32) - (((yy-y)/tf.cast(sigma,tf.float32))**2)/tf.cast(2,tf.float32))
		temp_heatmap = temp_heatmap * 255.
		temp_heatmap = temp_heatmap.numpy()
		temp_heatmap = np.reshape(temp_heatmap,(*r_output_shape,17))

		return temp_heatmap

	def gen_batch(self,batch_order):
		batch_imgs = []
		batch_heatmaps = []
		batch_valids = []
		b_start = self.start_idx[batch_order]
		b_end = self.end_idx[batch_order]
		temp_output_shape = self.output_shape
		temp_input_shape = self.input_shape
		temp_valids = self.valids
		temp_img_ids = self.img_ids
		temp_bbox = self.bbox
		temp_kps = self.scaled_kps
		temp_dict = self.id_to_file_dict



		for i in range(b_start,b_end):
			#valid
			i_valid = temp_valids[i]
			i_ones = np.ones((*temp_output_shape,17),dtype = np.float32)
			o_valid = i_ones*i_valid

			#heatmap
			i_kp = temp_kps[i]
			o_heatmap = self.render_gaussian_heatmap(i_kp,2)

			#imgs
			i_img_id = temp_img_ids[i]
			i_dir = temp_dict[i_img_id]
			i_dir = self.data_dir + i_dir
			o_img = mpimg.imread(i_dir)

			i_bbox = temp_bbox[i]
			o_crop = o_img[int(i_bbox[1]):int(i_bbox[1]+i_bbox[3]),int(i_bbox[0]):int(i_bbox[0]+i_bbox[2]),:]
			o_crop = resize(o_crop,temp_input_shape)
			o_crop = o_crop.astype('float32')

			batch_imgs.append(o_crop)
			batch_heatmaps.append(o_heatmap)
			batch_valids.append(o_valid)
		batch_imgs = np.array(batch_imgs)
		batch_heatmaps = np.array(batch_heatmaps)
		batch_valids = np.array(batch_valids)

		return batch_imgs, batch_heatmaps, batch_valids

	def shuffle_order(self):
		temp_bbox = self.bbox
		temp_kps = self.kps
		temp_scaled_kps = self.scaled_kps
		temp_track_ids = self.track_ids
		temp_img_ids = self.img_ids
		temp_valids = self.valids

		to_shuffle = list(zip(temp_bbox,temp_kps,temp_scaled_kps,temp_track_ids,temp_img_ids,temp_valids))
		random.shuffle(to_shuffle)
		temp_bbox,temp_kps,temp_scaled_kps,temp_track_ids,temp_img_ids,temp_valids = zip(*to_shuffle)

		self.bbox = temp_bbox
		self.kps = temp_kps
		self.scaled_kps = temp_scaled_kps
		self.track_ids = temp_track_ids
		self.img_ids = temp_img_ids
		self.valids = temp_valids
		