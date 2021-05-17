import numpy as np

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import gc

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))


def l0_distance(a, b, cutoff = 0, norm_bounds = None):
	dist = a-b
	dist[abs(dist)<=cutoff] = 0
	return np.sum(abs(dist)>0, axis = (-1,-2,-3))

def l1_distance(a, b, cutoff = 0, norm_bounds = None):
	dist = a-b
	dist[abs(dist)<=cutoff] = 0
	if norm_bounds is None:
		return np.sum(np.abs(dist), axis = (-1,-2,-3))
	else:
		max_norm = np.abs(norm_bounds[1] - norm_bounds[0])
		return np.mean(np.abs(dist), axis = (-1,-2,-3))/max_norm

def l2_distance(a, b, cutoff = 0, norm_bounds = None):
	dist = a-b
	dist[abs(dist)<=cutoff] = 0
	if norm_bounds is None:
		return np.sqrt(np.sum(dist*dist, axis = (-1,-2,-3)))
	else:
		max_norm = dist.size*(norm_bounds[1] - norm_bounds[0])**2 
		return np.sum(dist*dist, axis = (-1,-2,-3))/max_norm

def linf_distance(a, b, cutoff = 0, norm_bounds = None):
	dist = a-b
	dist[abs(dist)<=cutoff] = 0
	if norm_bounds is None:
		return np.max(np.abs(dist), axis = (-1,-2,-3))
	else:
		max_norm = np.abs(norm_bounds[1] - norm_bounds[0])
		return np.max(np.abs(dist), axis = (-1,-2,-3)) / max_norm

if __name__ == '__main__':
	pass