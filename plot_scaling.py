import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.optimize import curve_fit

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':9})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def get_ave_start_norm(n, norm):
	if norm == 0:
		return n
	elif norm == 1:
		return n*0.5
	elif norm == 2:
		return np.sqrt(n/3)
		# a = []
		# for i in n:
		# 	print(i)
		# 	vec = np.random.uniform(size = [1000,int(i)])
		# 	vec = np.sqrt(np.sum(vec*vec, axis = 1))
		# 	a.append(np.mean(vec))
		# return np.asarray(a)
	elif norm == -1:
		return 1.


def fit_given_norm(x,y, norm):
	# norm choices: 0 for l0 norm
	#				1 for l1 norm
	#				2 for l2 norm
	#				-1 for linf norm

	def func_sqrt_inv(x, c, x0):
		return c*(x - x0)**(-0.5)

	def func_constant(x, c, x0):
		return c*np.ones(x.shape)

	def func_sqrt(x, c, x0):
		return c*(x - x0)**(0.5)

	if norm == 0:
		func_choice= func_sqrt
	elif norm == 1: 
		func_choice= func_sqrt
	elif norm == 2:
		func_choice= func_constant
	elif norm == -1:
		func_choice= func_sqrt_inv
	else:
		raise ValueError('Not a valid norm choice: select from {-1,0,1,2}')

	non_nulls = np.logical_not(np.isnan(y))
	x = np.asarray(x).astype(np.float)
	y = np.asarray(y).astype(np.float)
	popt, _ = curve_fit(func_choice, x[non_nulls], y[non_nulls])

	return func_choice, popt


def get_norm_choice_str(norm):
	if norm == 0:
		return 'L0'
	elif norm == 1: 
		return 'L1'
	elif norm == 2:
		return 'L2'
	elif norm == -1:
		return 'Linf'
	else:
		raise ValueError('Not a valid norm choice: select from {-1,0,1,2}')

def get_norm_color(norm):
	if norm == 0:
		return 'c'
	elif norm == 1: 
		return 'b'
	elif norm == 2:
		return 'g'
	elif norm == -1:
		return 'm'
	else:
		raise ValueError('Not a valid norm choice: select from {-1,0,1,2}')

def plot_line_and_fit(x, y, y_up, y_down, norm, ax, normalized = False):

	y_err = np.zeros((2, len(x)))
	if normalized:
		y_err[0,:] = (y-y_down)/get_ave_start_norm(x,norm)
		y_err[1,:] = (y_up-y)/get_ave_start_norm(x,norm)		
	else:
		y_err[0,:] = y-y_down
		y_err[1,:] = y_up-y

	if normalized:
		ax.errorbar(x,y/get_ave_start_norm(x, norm), yerr = y_err,
			marker='.', markersize = 5., color=get_norm_color(norm), 
			linestyle = '', linewidth=0, alpha = 1., label = '_nolegend_', elinewidth = 1.)
	else:
		ax.errorbar(x,y, yerr = y_err, marker='.', markersize = 5., color='black', 
			linestyle = '', linewidth=0, alpha = 1., elinewidth = 1.)
	fit_fun, fit_data = fit_given_norm(x,y, norm = norm)
	x_fit = np.linspace(np.min(x), np.max(x), 200).reshape(-1)
	y_fit = fit_fun(x_fit, *fit_data)
	if normalized:
		ax.plot(x_fit, fit_fun(x_fit, *fit_data)/get_ave_start_norm(x_fit, norm), marker='', markersize = 0.0, 
			linestyle = '-', linewidth=1.5, alpha = 1., color = get_norm_color(norm))
	else:
		ax.plot(x_fit, fit_fun(x_fit, *fit_data), marker='', markersize = 0.0, color='red', 
			linestyle = '-', linewidth=1.5, alpha = 1.)

	ax.set_ylim(ymin=0)	
	# ax.text(.95,.9,get_norm_choice_str(norm),
	# 	horizontalalignment='right',
	# 	transform=ax.transAxes)

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

	if normalized:
		pass
	else:
		ax.legend(	['fit: '+fit_latex_str, 'data'],
				bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           		ncol=1, borderaxespad=0., prop={'size': 8})

	if normalized:
		pass
		# for spine in ax.spines.values():
		# 	spine.set_visible(False)
	else:
		ax.set_title(r'$\ell^{}$ norm'.format(norm_latex_str), y=-0.01)
		ax.set_ylabel('Median Distance '+r"$\Vert \Delta x \Vert_{}$".format(norm_latex_str))
		ax.set_xlabel('Input Dimension '+r"$n$")

def build_and_save_plot(df, filename):

	# style
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	# palette = plt.get_cmap('Set1')

	# fig, axes = plt.subplots( 1, 3 , sharex = True, sharey = True, gridspec_kw={'wspace': 0.05})
	fig, axes = plt.subplots( 1, 3, sharex = True) # gridspec_kw={'wspace': 0.03, 'hspace': 0.03}

	for i, norm in enumerate([1,2,-1]): #[0,1,2,-1] are the choices
		# current_ax = axes[int(i/2), i%2]
		current_ax = axes[i]
		norm_str = get_norm_choice_str(norm)
		plot_line_and_fit(	df.loc[df['distance_setting'] == norm_str, 'num_features'], 
							df.loc[df['distance_setting'] == norm_str, '50'],
							df.loc[df['distance_setting'] == norm_str, '55'],
							df.loc[df['distance_setting'] == norm_str, '45'], 
							norm, current_ax)


	fig.set_size_inches(6,3)
	plt.tight_layout()
	plt.savefig("./figures/"+filename, bbox_inches='tight')	
	plt.close()


def build_and_save_plot_normalized(df, filename):
	# style
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	# palette = plt.get_cmap('Set1')

	# fig, axes = plt.subplots( 1, 3 , sharex = True, sharey = True, gridspec_kw={'wspace': 0.05})
	fig, axes = plt.subplots( 1, 1) # gridspec_kw={'wspace': 0.03, 'hspace': 0.03}

	for i, norm in enumerate([1,2,-1]):
		current_ax = axes
		norm_str = get_norm_choice_str(norm)
		plot_line_and_fit(	df.loc[df['distance_setting'] == norm_str, 'num_features'], 
							df.loc[df['distance_setting'] == norm_str, '50'],
							df.loc[df['distance_setting'] == norm_str, '55'],
							df.loc[df['distance_setting'] == norm_str, '45'], 
							norm, current_ax, normalized = True)

	lgd = axes.legend([r'$\ell^1$ norm',r'$\ell^2$ norm', r'$\ell^{\infty}$ norm'], 
						loc='upper right',
           				ncol=1, borderaxespad=0., prop={'size': 8})
	axes.set_ylabel(r'Relative Distance $\frac{\Vert \Delta x \Vert}{\Vert x_0 \Vert}$')
	axes.set_xlabel(r'Input Dimension $n$')
	# fig.subplots_adjust(right=0.7)
	plt.autoscale()
	plt.tight_layout()


	fig.set_size_inches(4,3)
	plt.savefig("./figures/"+filename, bbox_inches = 'tight', bbox_extra_artists=(lgd,))	


	plt.close()



if __name__ == '__main__':

	df = pd.read_csv('./figures/figure_data/resnet_random.csv')
	build_and_save_plot(df, 'resnet_random_absolute.pdf')
	build_and_save_plot_normalized(df, 'resnet_random_normalized.pdf')

	df = pd.read_csv('./figures/figure_data/lenet_random.csv')
	build_and_save_plot(df, 'lenet_random_absolute.pdf')
	build_and_save_plot_normalized(df, 'lenet_random_normalized.pdf')

	df = pd.read_csv('./figures/figure_data/CNN_random.csv')
	build_and_save_plot(df, 'CNN_random_absolute.pdf')
	build_and_save_plot_normalized(df, 'CNN_random_normalized.pdf')

	df = pd.read_csv('./figures/figure_data/FCN_random.csv')
	build_and_save_plot(df, 'FCN_random_absolute.pdf')
	build_and_save_plot_normalized(df, 'FCN_random_normalized.pdf')

