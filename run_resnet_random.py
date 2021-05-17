import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"    

import keras
import foolbox
from setup_model_img import FF_img_model, img_data_generator, simple_conv_model, VGG_img_model
from foolbox_attack_setup import 	distance_objects, distances_names, \
									attack_dict_list
import numpy as np
import utils
import pandas as pd
import time
from setup_resnet import resnet_v1

import tensorflow as tf


def initialize_df():
	columns = [	'simulation_number',
				'image_size',
				'number_labels',
				'attack_name',
				'distance_setting',
				'model_number',
				'image_id',
				'time_of_simulation',
				'attack_failure',
				'start_val_label0',
				'start_val_label1',
				'l0',
				'l1',
				'l2',
				'linf']

	df = pd.DataFrame(columns = columns)
	return df

def append_pandas(df):
	new_data_dict = {	
			'simulation_number': simulation_number,
			'image_size': image_size,
			'number_labels': num_labels,
			'attack_name': attack_name,
			'distance_setting': distance_name,
			'model_number': model_i,
			'image_id': image_ids,
			'time_of_simulation': time_of_simulation,
			'attack_failure': attack_failed,
			'start_val_label0': raw_labels[:,0],
			'start_val_label1': raw_labels[:,1],
			'l0': l0_distances,
			'l1': l1_distances,
			'l2': l2_distances,
			'linf': linf_distances
			}

	df = pd.DataFrame.from_dict(new_data_dict)
	# df = df.append( df )

	return df


bounds=(0, 1)
num_labels = 2
simulation_number = 1
n_test = 10
n_models = 300
n_channels = 3

csv_name = 'norm_scaling_resnet20_1.csv'
df = initialize_df()
df.to_csv('./csv_files/' + csv_name)


for image_size in range(12,61,4):
	for distance_type, distance_name, all_attacks in zip(distance_objects, distances_names, attack_dict_list):

		for model_i in range(n_models):

			utils.reset_keras()

			input_shape = [image_size, image_size, 3]

			model = resnet_v1(input_shape, 20)

			datagen = img_data_generator(num_labels = num_labels, 
										 image_size = image_size, 
										 bounds = bounds, 
										 num_channels = 3)

			images, labels = datagen.generate_dataset(n_test)
			image_ids = list(range(n_test))
			print(images.shape)

			raw_labels = model.model.predict(images)
			labels = model.model.predict(images).argmax(axis=-1)	# override labels to true initial values

			fmodel = foolbox.models.KerasModel(	model.model, 
												bounds=bounds, 
												# num_classes=num_labels,
												channel_axis = 3,
												predicts='logits')


			for attack_dict in all_attacks:
				attack_name = attack_dict['name']
				f_attack = attack_dict['object']
				attack_params = attack_dict['params']
				# 	print(distance_name)
				print(attack_name)

				start_time =  time.time()

				try:
					attack = f_attack(fmodel, distance = distance_type)
					adversarials = attack(images, labels, unpack = False, **attack_params)

					# distances = np.asarray([a.distance.value for a in adversarials])
					# print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))
					# print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))
					# print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))

					l0_distances = np.asarray([utils.l0_distance(a.perturbed, a.unperturbed) for a in adversarials])
					l1_distances = np.asarray([utils.l1_distance(a.perturbed, a.unperturbed) for a in adversarials])
					l2_distances = np.asarray([utils.l2_distance(a.perturbed, a.unperturbed) for a in adversarials])
					linf_distances = np.asarray([utils.linf_distance(a.perturbed, a.unperturbed) for a in adversarials])
					attack_failed = [a.distance.value == np.inf for a in adversarials]
					attack_success = True
				except Exception as e:
					print(e)
					print('failed')

					l0_distances = [None]*labels.size
					l1_distances = [None]*labels.size
					l2_distances = [None]*labels.size
					linf_distances = [None]*labels.size
					attack_failed = [True]*n_test
					attack_success = False					


				print()

				time_of_simulation = time.time()-start_time

				if attack_success:
					df = append_pandas(df)
					df.to_csv('./csv_files/' + csv_name, mode='a', header=False)

				simulation_number += 1

			
