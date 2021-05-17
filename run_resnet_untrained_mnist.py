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
from dataset_utils import load_dataset, accuracy_from_real
import tensorflow as tf


def initialize_df():
	columns = [	'simulation_number',
				'image_size',
				'number_labels',
				'attack_name',
				'distance_setting',
				'model_number',
				'train_accuracy',
				'test_accuracy',
				'image_id',
				'image_type',
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
			'train_accuracy': train_accuracy,
			'test_accuracy': test_accuracy,
			'image_id': image_ids,
			'image_type': image_type,
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
n_test = 20
n_models = 250

csv_name = 'resnet20untrained_noscaledmnist_1.csv'
df = initialize_df()
df.to_csv('./csv_files/' + csv_name)

distance_objects = distance_objects[:-1]
distances_names = distances_names[:-1]
attack_dict_list = attack_dict_list[:-1]

image_size = 28

(X_train, y_train), (X_test, y_test) = load_dataset('mnist', list(range(10)), [0,1]*5,
													train_size_subset = None, 
													test_size_subset = None, 
													flatten_images = False,
													normalize_images = False,
													divide_by_255 = True, 
													pytorch_format_2d = False,
													new_size = None,
													convert_y_to_int = True)

datagen = img_data_generator(num_labels = num_labels, 
						 image_size = image_size, 
						 bounds = bounds, 
						 num_channels = 1)

image_ids = list(range(n_test))

for distance_type, distance_name, all_attacks in zip(distance_objects, distances_names, attack_dict_list):

	for model_i in range(n_models):
		utils.reset_keras()

		input_shape = [image_size, image_size, 1]
		model = resnet_v1(input_shape, 20, use_logits = False)
		y_train_vals = model.predict(X_train)
		y_test_vals = model.predict(X_test)
		train_accuracy = accuracy_from_real(y_train_vals, y_train)
		test_accuracy = accuracy_from_real(y_test_vals, y_test)
		print(train_accuracy)
		print(test_accuracy)
		

		fmodel = foolbox.models.KerasModel(	model.model, 
											bounds=bounds, 
											# num_classes=num_labels,
											channel_axis = 3,
											predicts='probabilities')

		X_train_attack = X_train[np.random.choice(len(y_train), n_test)]
		X_test_attack = X_test[np.random.choice(len(y_test), n_test)]
		X_random_attack, labels = datagen.generate_dataset(n_test)


		for image_type, images in zip(['train','test','random'], [X_train_attack, X_test_attack, X_random_attack]):
			raw_labels = model.model.predict(images)
			labels = model.model.predict(images).argmax(axis=-1)	# override labels to true initial values

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

			
