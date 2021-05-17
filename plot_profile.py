import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.optimize import curve_fit
from plot_scaling import fit_given_norm, get_norm_choice_str

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':9})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



def plot_single(y,ax,norm,dataset_name, include_linear_fit = True):
	for image_type in ['random','train','test']:
		y_plot = y.loc[y['image_type'] == image_type,'distance']
		x_plot = np.linspace(0,1,len(y_plot))
		ax.plot(x_plot,np.sort(y_plot), '.',markersize = 0.2)

	if norm == -1:
		norm_latex_str = "{\infty}"
	elif norm == 0:
		norm_latex_str = "{0}"
	elif norm == 1:
		norm_latex_str = "{1}"
	elif norm == 2:
		norm_latex_str = "{2}"

	if include_linear_fit:
		y_plot = np.sort(y.loc[y['image_type'] == 'random','distance'])
		x_plot = np.linspace(0,1,len(y_plot))
		m,b = np.polyfit(x_plot[:int(len(y_plot)/4)],y_plot[:int(len(y_plot)/4)], 1)
		ax.plot(x_plot, m*x_plot+b,'--',linewidth=0.5,alpha = 0.5,color = 'blue')

	ax.set_title(dataset_name)
	ax.set_ylabel('Adversarial Distance '+r"$\Vert \Delta x \Vert_{}$".format(norm_latex_str))
	ax.set_xlabel('Percentile')
	lgnd = ax.legend(['random images','train set images', 'test set images'])
	lgnd.legendHandles[0]._legmarker.set_markersize(6)
	lgnd.legendHandles[1]._legmarker.set_markersize(6)
	lgnd.legendHandles[2]._legmarker.set_markersize(6)


def build_and_save_plot(df_mnist, df_cifar, filename, norm = 1):

	# style
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	# palette = plt.get_cmap('Set1')

	fig, axes = plt.subplots( 1, 2, sharex = True) # gridspec_kw={'wspace': 0.03, 'hspace': 0.03}
	categories = ['MNIST','CIFAR10']
	im_sizes = [28*28,32*32*3]

	norm_str = get_norm_choice_str(norm)

	for ax, df, dataset_name, im_size in zip(axes, [df_mnist, df_cifar], categories, im_sizes): 
		plot_single(df[ (df['distance_setting']==norm_str) 
						& (df['num_features']== im_size) ], 
					ax,
					norm,
					dataset_name)




	# fig = plt.gcf()
	fig.set_size_inches(5.5,3)
	plt.tight_layout()
	plt.savefig("./figures/"+filename, bbox_inches='tight')	
	plt.close()



if __name__ == '__main__':

	df_mnist = pd.read_csv('./figures/figure_data/resnet_untrained_noscaled_mnist_full.csv')
	df_cifar = pd.read_csv('./figures/figure_data/resnet_untrained_noscaled_cifar10_full.csv')


	build_and_save_plot(df_mnist,df_cifar, 'resnet_untrained_profiles.pdf')

