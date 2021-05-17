import pandas as pd
import glob
import numpy as np

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

path = r'./csv_files' # use your path



def concatenate_and_save_csv(input_file_string, final_file_name, summarized = True, is_trained = False, is_MNIST = False):
    # Example:
    # csv_file_id = "*_resnet20trained_*mnist*"
    # final_file_name = 'resnet20_trained_mnist.csv'
    # is_trained = True
    # is_MNIST = True

    final_file_name = 'figures/figure_data/' + final_file_name

    all_files = glob.glob(path + "/" + input_file_string + ".csv")

    li = []

    for i, filename in enumerate(all_files):
        print(filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        df['file_number'] = i
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)


    print(frame.columns)

    frame['relevant_norm'] = None
    frame.loc[frame['distance_setting'] == 'L0','relevant_norm'] = frame.loc[frame['distance_setting'] == 'L0','l0']
    frame.loc[frame['distance_setting'] == 'L1','relevant_norm'] = frame.loc[frame['distance_setting'] == 'L1','l1']
    frame.loc[frame['distance_setting'] == 'L2','relevant_norm'] = frame.loc[frame['distance_setting'] == 'L2','l2']
    frame.loc[frame['distance_setting'] == 'Linf','relevant_norm'] = frame.loc[frame['distance_setting'] == 'Linf','linf']

    if is_MNIST:
    	frame['num_features'] = frame['image_size']*frame['image_size']
    else:
    	frame['num_features'] = frame['image_size']*frame['image_size']*3

    if is_trained:
        agg_data = frame.groupby(['num_features','file_number','image_id','distance_setting','model_number','image_type'
                                ]).agg( {
                                            'relevant_norm':'min'
                                        })
        if summarized:
            agg_data = agg_data.groupby(['num_features','distance_setting','image_type']).agg({'relevant_norm':['median', percentile(40), percentile(45), percentile(55), percentile(60)]})


    else:
        agg_data = frame.groupby(['num_features','file_number','image_id','distance_setting','model_number'
        						]).agg( {
        									'relevant_norm':'min'
        								})
        if summarized:
            agg_data = agg_data.groupby(['num_features','distance_setting']).agg({'relevant_norm':['median', percentile(40), percentile(45), percentile(55), percentile(60)]})



    print(agg_data)
    # print(agg_data.columns)    
    agg_data = pd.DataFrame(agg_data.to_records())
    if summarized:
        if is_trained:
            agg_data.columns = ['num_features','distance_setting','image_type','50','40','45', '55','60']
        else:
            agg_data.columns = ['num_features','distance_setting','50','40','45', '55','60']

    else:
        if is_trained:
            agg_data.columns = ['num_features','file_number','image_id','distance_setting','model_number','image_type','distance']
        else:
            agg_data.columns = ['num_features','file_number','image_id','distance_setting','model_number','distance']        

    agg_data.to_csv(final_file_name)



if __name__ == '__main__':

    # for each of the files below, one must first make the raw data necessary to construct the right data for the figures - see readme
    # you may comment out rows if you have not yet created the requisite file

    # first argument is the matching string for any files that are for the given summarized file
    # Second argument is the name used to save the resulting summarized data
    
    concatenate_and_save_csv("*norm_scaling_resnet20_*", 'resnet_random.csv', is_trained = False)
    concatenate_and_save_csv("*norm_scaling_lenet_*", 'lenet_random.csv', is_trained = False)
    concatenate_and_save_csv("*norm_scaling_simplecnn_*", 'CNN_random.csv', is_trained = False)
    concatenate_and_save_csv("*norm_scaling_fcn_*", 'FCN_random.csv', is_trained = False)


    concatenate_and_save_csv("*resnet20trained_noscaledmnist*", 'resnet_trained_noscaled_mnist.csv', is_trained = True, is_MNIST = True)
    concatenate_and_save_csv("*resnet20trained_noscaledcifar10*", 'resnet_trained_noscaled_cifar10.csv', is_trained = True, is_MNIST = False)
    concatenate_and_save_csv("*resnet20untrained_noscaledmnist*", 'resnet_untrained_noscaled_mnist.csv', is_trained = True, is_MNIST = True)
    concatenate_and_save_csv("*resnet20untrained_noscaledcifar10*", 'resnet_untrained_noscaled_cifar10.csv', is_trained = True, is_MNIST = False)
    concatenate_and_save_csv("*resnet20trained_noscaledmnist*", 'resnet_trained_noscaled_mnist_full.csv', summarized = False, is_trained = True, is_MNIST = True)
    concatenate_and_save_csv("*resnet20trained_noscaledcifar10*", 'resnet_trained_noscaled_cifar10_full.csv', summarized = False, is_trained = True, is_MNIST = False)
    concatenate_and_save_csv("*resnet20untrained_noscaledmnist*", 'resnet_untrained_noscaled_mnist_full.csv', summarized = False, is_trained = True, is_MNIST = True)
    concatenate_and_save_csv("*resnet20untrained_noscaledcifar10*", 'resnet_untrained_noscaled_cifar10_full.csv', summarized = False, is_trained = True, is_MNIST = False)
