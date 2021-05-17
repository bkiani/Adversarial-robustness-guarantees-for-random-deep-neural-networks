
# Code for paper "Adversarial robustness guarantees for random deep neural networks"

Code to replicate figures and results in our paper "Adversarial robustness guarantees for random deep neural networks".

## Getting Started

The code is written in Python. Adversarial attacks on neural networks are performed using the python package Foolbox. For the code to work, you will need to install various packages including the ones below. Note, version used in our computation for certain packages is given in parantheses (code still may work with different versoins).

```
Foolbox 
Numpy (1.16.4)
Keras (2.2.4)
Tensorflow (1.14.0)
Matplotlib
Pandas
Scipy
```

For installing packages, we used the program Anaconda (https://www.anaconda.com/). 

We also highly recommend performing computation on a GPU; computations take a long time to complete and may not be feasible with a CPU.

## Performing Simulations

Files for performing attacks on random or trained networks are listed below. Each outputs raw data to a csv file whose name can be controlled by the csv_name variable. All the files for running simulations begin with the run_* prefix.

1. run_resnet_random: (figure 1,2) attacks random resnet style networks and outputs raw data to csv file specified
2. four files need to be run for figures 3 and 4. These correspond to files for outputting raw data for untrained CIFAR/MNIST and trained CIFAR/MNIST networks
	1. run_resnet_trained_cifar10: attacking resnet style networks trained on CIFAR10 
	2. run_resnet_untrained_cifar10: attacking resnet style networks of the same form as prior file but untrained (random)
	3. run_resnet_trained_mnist: attacking resnet style networks trained on MNIST
	4. run_resnet_untrained_mnist: attacking resnet style networks of the same form as prior file but untrained (random)
3. run_lenet_random: (figure 1,2 in supplementary) attacks random lenet style networks and outputs raw data to csv file specified
4. run_fcn_random: (figure 3,4 in supplementary) attacks random fully connected networks and outputs raw data to csv file specified
5. run_simple_cnn_random: (figure 5,6 in supplementary) attacks random networks with convolutional layers and outputs raw data to csv file specified

All raw data for the above code is outputted to the csv_files directory.

For any code which uses a GPU, at the top of the file is a parameter which can control which GPU is used (os.environ["CUDA_VISIBLE_DEVICES"]). Set this to "0" to use the first or default GPU.

## Making Figures

Raw data for all figures are contained in csv files which can be made using the code described in the prior section.

Before making any figures, run the analyze_csv.py file to summarize the data (e.g. calculate medians) and format it in preparation for plotting. Summarized used for plots is contained in the figures/figure_data directory.

There are three plotting files:
1. plot_scaling: outputs plots for random neural networks (e.g. figures 1 and 2)
2. plot_scaling_trained: outputs a plot comparing trained to untrained neural networks (figure 3) 
3. plot_profile: outputs a plot of the adversarial distance by percentile (figure 4)

All figures are outputted to the figures folder. Example figures are shown there.

## Authors

* Giacomo De Palma (Scuola Normale Superiore)
* [Bobak Kiani](https://github.com/bkiani) (MIT) 
* Seth Lloyd (MIT)

