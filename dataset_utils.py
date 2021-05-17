import numpy as np
import statistics
import pandas as pd
import pickle
import random
import time
import h5py
from keras.datasets import mnist, cifar10
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt


# get accuracy of model
def accuracy_from_real(pred, actual):
	pred = np.argmax(pred, axis=1)
	n_correct = np.count_nonzero(pred == actual)
	return (n_correct)/len(actual)



def load_dataset(set_name = 'mnist', 
	digits = list(range(10)), 
	classifications = [-1.,1.]*5, 
	train_size_subset = None, 
	test_size_subset = None,
	new_size = None, 
	flatten_images = True,
	normalize_images = True, 
	pytorch_format_2d = False,
	subtract_by_mean = True,
	convert_y_to_int = False,
	divide_by_255 = True):
	"""Loads data from a pickle file.
	"""

	# load data from pickle
	if set_name == 'cifar10':
		(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	elif set_name == 'mnist':
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
	else:
		raise ValueError('Not a valid dataset name')


	# fix cifar10 outputs to 1d vectors
	if set_name == 'cifar10':
		y_train = y_train.reshape(len(y_train))
		y_test = y_test.reshape(len(y_test))

	# add single color channel to mnist
	if set_name == 'mnist':
		X_train = X_train.reshape(list(X_train.shape) + [1])
		X_test = X_test.reshape(list(X_test.shape) + [1])

	# only select digits which are specified in input
	X_train = X_train[np.isin(y_train, digits)]
	y_train = y_train[np.isin(y_train, digits)]
	X_test = X_test[np.isin(y_test, digits)]
	y_test = y_test[np.isin(y_test, digits)]

	# get a random subset of the full dataset
	if train_size_subset is not None:
		subset = random.sample(list(range(X_train.shape[0])), train_size_subset)
		X_train = X_train[subset]
		y_train = y_train[subset]

	# get a random subset of the full dataset
	if test_size_subset is not None:
		subset = random.sample(list(range(X_test.shape[0])), test_size_subset)
		X_test = X_test[subset]
		y_test = y_test[subset]

	if flatten_images:
		# flatten 2d images to 1d format
		X_train = flatten_img(X_train)
		X_test = flatten_img(X_test)
	else:
		if pytorch_format_2d:
			X_train = reshape_channels(X_train)
			X_test = reshape_channels(X_test)


	if new_size is not None:
		X_train = resize_img(X_train, [X_train.shape[0]]+new_size+[X_train.shape[3]])
		X_test = resize_img(X_test, [X_test.shape[0]]+new_size+[X_test.shape[3]])

	if divide_by_255:
		X_train = X_train / 255
		X_test = X_test / 255
	
	if normalize_images:
		# normalize image values
		X_train = normalize_img(X_train, subtract_by_mean = subtract_by_mean)
		X_test = normalize_img(X_test, subtract_by_mean = subtract_by_mean)


	# placeholder for new classifications
	y_train_class = np.ones(y_train.shape)*(-1)
	y_test_class = np.ones(y_test.shape)*(-1)
	
	# give specified classifications
	for digit_i, label_i in zip(digits, classifications):
		y_train_class[y_train == digit_i] = label_i
		y_test_class[y_test == digit_i] = label_i

	if convert_y_to_int:
		y_train_class = y_train_class.astype(int)
		y_test_class  = y_test_class.astype(int)

	return [(X_train, y_train_class), (X_test, y_test_class)]

def flatten_img(X_in):
	X_shape = X_in.shape
	X_out = X_in.reshape(X_shape[0], np.prod(X_shape[1:]))
	return X_out

def reshape_channels(X_in):
	X_shape = X_in.shape
	X_out = np.transpose(X_in, (0,3,1,2))
	return X_out

def normalize_img(X_in, norm = None, subtract_by_mean = True):
	if norm is None:
		norm = np.prod(X_in.shape[1:])
	if subtract_by_mean:
		X_means = np.mean(X_in.astype(float), axis = tuple(range(1,len(X_in.shape))))
		X_in = X_in.transpose()
		X_in = X_in - X_means
		X_in = X_in.transpose()
	X_lengths = np.sqrt(np.sum(X_in.astype(float)**2, axis = tuple(range(1,len(X_in.shape))) )/norm)
	X_out = X_in.transpose()/X_lengths
	return X_out.transpose()

def resize_img(X_in, new_size):
	return resize(X_in, new_size, anti_aliasing = False) 


if __name__ == '__main__':
	(X_train, y_train), (X_test, y_test) = load_dataset('cifar10', list(range(10)), [0,1]*5,
														train_size_subset = None, 
														test_size_subset = None, 
														flatten_images = False,
														normalize_images = False,
														divide_by_255 = False, 
														pytorch_format_2d = False,
														new_size = None,
														convert_y_to_int = True)

	print(np.max(X_train))
	print(X_train[0])
	print(X_train.shape)
	# print(X_train_2)

	# fig, axes = plt.subplots(nrows=2, ncols=2)

	# ax = axes.ravel()

	# ax[0].imshow(X_train[0].reshape(X_train[0].shape[:2]), cmap='gray')
	# ax[0].set_title("Original image 1")

	# ax[1].imshow(X_train_2[0].reshape(X_train_2[0].shape[:2]), cmap='gray')
	# ax[1].set_title("Resized image 1")

	# ax[2].imshow(X_train[1].reshape(X_train[0].shape[:2]), cmap='gray')
	# ax[2].set_title("Original image 2")

	# ax[3].imshow(X_train_2[1].reshape(X_train_2[0].shape[:2]), cmap='gray')
	# ax[3].set_title("Resized image 2")

	# # ax[0].set_xlim(0, 512)
	# # ax[0].set_ylim(512, 0)
	# plt.tight_layout()
	# plt.show()

	# # plt.savefig('test.png')