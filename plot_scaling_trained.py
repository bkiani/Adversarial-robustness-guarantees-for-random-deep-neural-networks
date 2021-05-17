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


def get_relevant_data(df, image_type, norm_str, im_size):
	return df.loc[ (df['distance_setting']==norm_str) 
					& (df['num_features']== im_size)
					& (df['image_type'] == image_type),
					['45','50','55'] ]


def build_and_save_plot(df_mnist, df_cifar, df_mnist_untrained, df_cifar_untrained, filename, norm = 1):

	# style
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	# palette = plt.get_cmap('Set1')

	# fig, axes = plt.subplots( 1, 3, sharex = True) # gridspec_kw={'wspace': 0.03, 'hspace': 0.03}
	categories = [	'MNIST (untrained)',
					'MNIST (trained)',
					'CIFAR10 (untrained)', 
					'CIFAR10 (trained)']

	data_list =  [	df_mnist_untrained,
					df_mnist,
					df_cifar_untrained,
					df_cifar
					]

	im_sizes = [28*28,32*32*3]

	data_sizes = [im_sizes[0]]*2 + [im_sizes[1]]*2


	norm_str = get_norm_choice_str(norm)

	for i, image_type in enumerate(['random','train','test']): #[0,1,2,-1] are the choices
		relevant_data = [get_relevant_data(i,image_type,norm_str, i_size) for i, i_size in zip(data_list,data_sizes)]

		plot_data = [  df['50'].iloc[0] \
						for df in relevant_data ]

		err = [ [df['55'].iloc[0]-df['50'].iloc[0], df['55'].iloc[0]-df['50'].iloc[0]] \
					for df in relevant_data ]
		err= np.asarray(err).T


		plt.errorbar(	plot_data,
						categories,
						xerr = err,
						fmt = 'o',
						markersize = 5.,
						linewidth = 0,
						elinewidth = 2,
						alpha = 0.5
					)

	if norm == -1:
		norm_latex_str = "{\infty}"
		# fit_latex_str = r"$\Vert x_{} - x_0 \Vert_{}=\frac{}{}$".format("{adv}",norm_latex_str,"{C}", "{\sqrt{n-n_0}}")
		fit_latex_str = r"$\Vert \Delta x \Vert_{} \propto \frac{}{}$".format(norm_latex_str,"{1}","{\sqrt{n}}")
	elif norm == 0:
		norm_latex_str = "{0}"
		# fit_latex_str = r"$\Vert x_{} - x_0 \Vert_{}= {} {}$".format("{adv}",norm_latex_str,"C", "\sqrt{n-n_0}")
		fit_latex_str = r"$\Vert \Delta x \Vert_{} \propto \sqrt{}$".format(norm_latex_str,"{n}")
	elif norm == 1:
		norm_latex_str = "{1}"
		# fit_latex_str = r"$\Vert x_{} - x_0 \Vert_{}= {} {}$".format("{adv}",norm_latex_str,"C", "\sqrt{n-n_0}")
		fit_latex_str = r"$\Vert \Delta x \Vert_{} \propto \sqrt{}$".format(norm_latex_str,"{n}")
	elif norm == 2:
		norm_latex_str = "{2}"
		# fit_latex_str = r"$\Vert x_{} - x_0 \Vert_{}= {}$".format("{adv}",norm_latex_str,"C")
		fit_latex_str = r"$\Vert \Delta x \Vert_{}= {}$".format(norm_latex_str,"C")

	plt.gca().set_xlim(left=0)

	plt.xlabel('Median Distance '+r"$\Vert \Delta x \Vert_{}$".format(norm_latex_str))
	plt.legend(['random images','train set', 'test set'])
	fig = plt.gcf()
	fig.set_size_inches(5,2)
	plt.tight_layout()
	plt.savefig("./figures/"+filename, bbox_inches='tight')	
	plt.close()



if __name__ == '__main__':

	df_mnist = pd.read_csv('./figures/figure_data/resnet_trained_noscaled_mnist.csv')
	df_cifar = pd.read_csv('./figures/figure_data/resnet_trained_noscaled_cifar10.csv')
	df_mnist_untrained = pd.read_csv('./figures/figure_data/resnet_untrained_noscaled_mnist.csv')
	df_cifar_untrained = pd.read_csv('./figures/figure_data/resnet_untrained_noscaled_cifar10.csv')

	build_and_save_plot(df_mnist,df_cifar,
						df_mnist_untrained, df_cifar_untrained,
						'resnet_trained_comparison.pdf')
