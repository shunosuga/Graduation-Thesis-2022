import os
import bdpy
from bdpy.dataform import Features, save_array, load_array
from bdpy.ml import ModelTraining, ModelTest
from bdpy.util import get_refdata, makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np

# fMRI data for decoder training and test
fmri_root_dir   = '/home/share/data/fmri_shared/datasets/Deeprecon/fmriprep'

fmri_data_train = 'ES_ImageNetTraining_volume_native.h5'
fmri_data_test  = 'ES_ImageNetTest_volume_native.h5'
subject = 'ES'

rois = ['ROI_VC', 'ROI_V1', 'ROI_V2', 'ROI_V3', 'ROI_hV4', 'ROI_HVC']

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

# The number of voxels selected in automatic feature selection of Fastl2LiR
num_voxels = 500

feature_dir = '/home/shunosuga/data/features/u_net'
image_dir = '/home/shunosuga/data/image_deeprecon'

# Learning parameters
alpha = 100  # Regularization parameter of FastL2LiR
chunk_axis = 1  # DNN features are divided into chunks along this axis and trained separatedly.

# Directory settings
decoders_dir     = '/home/shunosuga/data/feature_decoders/deeprecon_unet'
decoded_feat_dir = '/home/shunosuga/data/decoded_features/deeprecon_unet'

# Load data
bdata_train = bdpy.BData(os.path.join(fmri_root_dir, fmri_data_train))


# Load test data
bdata_test = bdpy.BData(os.path.join(fmri_root_dir, fmri_data_test))


def predtest(roi, layer):
    x = bdata_train.select(roi)
    y = np.load(os.path.join(feature_dir, layer, 'train.npy'))

    x_labels = bdata_train.get_labels('stimulus_name')
    y_labels = np.load(os.path.join(image_dir, 'train_list_.npy'))

    # Normalize X and Y
    x_mean = np.mean(x, axis=0)[np.newaxis, :]
    x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]

    y_mean = np.mean(y, axis=0)[np.newaxis, :]
    y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]

    y_index = np.array([np.where(np.array(y_labels) == xl) for xl in x_labels]).flatten()

    # Model setup
    model = FastL2LiR()
    model_param = {
        'alpha':  alpha,
        'n_feat': num_voxels
    }

    # The directory to save the trained model
    model_dir = os.path.join(decoders_dir, 'caffenet', layer, subject, roi, 'model')
    makedir_ifnot(model_dir)
    
    # Save X, Y mean and norm
    save_array(os.path.join(model_dir, 'x_mean.mat'), x_mean, key='x_mean', dtype=np.float32, sparse=False)
    save_array(os.path.join(model_dir, 'y_mean.mat'), y_mean, key='y_mean', dtype=np.float32, sparse=False)
    save_array(os.path.join(model_dir, 'x_norm.mat'), x_norm, key='x_norm', dtype=np.float32, sparse=False)
    save_array(os.path.join(model_dir, 'y_norm.mat'), y_norm, key='y_norm', dtype=np.float32, sparse=False)

    train_id = 'feature-decoding-training_' + subject + '_' + roi + '_' + layer

    # Training
    train = ModelTraining(model, x, y)

    # Define training settings
    train.id = train_id
    train.model_parameters = model_param

    train.X_normalize = {
        'mean': x_mean,
        'std': x_norm
    }
    train.Y_normalize = {
        'mean': y_mean,
        'std': y_norm
    }
    train.Y_sort = {'index': y_index}

    train.dtype = np.float32
    train.chunk_axis = chunk_axis
    train.save_format = 'bdmodel'
    train.save_path = model_dir

    # Run training
    train.run()

    x = bdata_test.select(roi)
    y = np.load(os.path.join(feature_dir, layer, 'test.npy'))

    x_labels = bdata_test.get_labels('stimulus_name')
    y_labels = np.load(os.path.join(image_dir, 'test_label.npy'))

    # Average fMRI data in each 
    x_labels_unique = np.unique(x_labels)
    x = np.vstack([np.mean(x[(np.array(x_labels) == lb).flatten(), :], axis=0) for lb in x_labels_unique])

    model_dir = os.path.join(decoders_dir, 'caffenet', layer, subject, roi, 'model')

    output_dir_decoded_feat = os.path.join(decoded_feat_dir, 'decoded_features', 'caffenet', layer, subject, roi)
    output_dir_profile_corr = os.path.join(decoded_feat_dir, 'prediction_accuracy', 'caffenet', layer, subject, roi)

    makedir_ifnot(output_dir_decoded_feat)
    makedir_ifnot(output_dir_profile_corr)

    # Load mean and norm of training X and Y
    x_mean = load_array(os.path.join(model_dir, 'x_mean.mat'), key='x_mean')  # shape = (1, n_voxels)
    x_norm = load_array(os.path.join(model_dir, 'x_norm.mat'), key='x_norm')  # shape = (1, n_voxels)
    y_mean = load_array(os.path.join(model_dir, 'y_mean.mat'), key='y_mean')  # shape = (1, shape_features)
    y_norm = load_array(os.path.join(model_dir, 'y_norm.mat'), key='y_norm')  # shape = (1, shape_features)

    # Normalize X (fMRI data)
    x = (x - x_mean) / x_norm

    # Prediction
    model = FastL2LiR()

    test = ModelTest(model, x)
    test.model_format = 'bdmodel'
    test.model_path = model_dir
    test.dtype = np.float32
    test.chunk_axis = chunk_axis

    y_pred = test.run()

    # Postprocessing (denormalize Y)
    y_pred = y_pred * y_norm + y_mean

    # Save the decoded features
    for i, label in enumerate(x_labels_unique):
        feat = np.array([y_pred[i,]])  # To make feat shape 1 x M x N x ...

        save_file = os.path.join(output_dir_decoded_feat, '%s.mat' % label)

        save_array(save_file, feat, key='feat', dtype=np.float32, sparse=False)

    # Reshape true and predicted Y as 2-d arrays to calculate correlations easily
    y_pred_2d = y_pred.reshape([y_pred.shape[0], -1])
    y_true_2d = y.reshape([y.shape[0], -1])

    # Sort true Y along labels on predicted Y (x_labels_unique)
    y_true_2d = get_refdata(y_true_2d, np.array(y_labels), x_labels_unique)

    # Calculate profile correlation (and reshape)

    n_units = y_true_2d.shape[1]

    profile_correlation = np.array([np.corrcoef(y_pred_2d[:, i].flatten(), y_true_2d[:, i].flatten())[0, 1] for i in range(n_units)])
    profile_correlation = profile_correlation.reshape((1,) + y_pred.shape[1:])

    # Save results
    save_file = os.path.join(output_dir_profile_corr, 'profile_correlation.mat')
    save_array(save_file, profile_correlation, key='profile_correlation', dtype=np.float32, sparse=False)
    

for roi in rois:
    for layer in layers:
        predtest(roi, layer)






