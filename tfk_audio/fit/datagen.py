import os
from typing import Optional, Callable
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .labels import *


def make_spec_dataset(files: list,
                      image_shape: tuple,
                      batch_size: int = 1,
                      augment: bool = True,
                      augment_blend_prob: float = 0.5,
                      augment_max_time_masks: int = 5,
                      augment_max_time_mask_size: float = 0.1,
                      augment_max_freq_masks: int = 5,
                      augment_max_freq_mask_size: float = 0.1,
                      augment_add_noise_prob: float = 0.0,
                      augment_add_noise_stds: float = 0.5,
                      augment_max_time_shift: float = 0.33,
                      augment_max_freq_shift: float = 0.05,
                      augment_max_contrast: float = 2.0,
                      shuffle: bool = True):
    ''' Prepares a tf.data.Dataset for generating spectrogram training data
    
    Args:
        files:                       list of paths to tfrecord files
        image_shape:                 integer tuple of image shape: (height, width)
        batch_size:                  samples per batch
        augment:                     whether to apply data augmentation
        augment_max_time_masks:      max number of masks to apply in the time axis (https://arxiv.org/abs/1904.08779)
        augment_max_time_mask_size:  max desired mask width (fraction of total time)
        augment_max_freq_masks:      max number of masks to apply in the freq axis (https://arxiv.org/abs/1904.08779)
        augment_max_freq_mask_size:  max desired mask width (fraction of total frequency range)
        augment_add_noise_prob:      probability of adding noise to a sample
        augment_add_noise_stds:      standard deviation of added Gaussian noise (# of batch standard deviations)
        augment_max_time_shift:      max time shift (fraction of total time)
        augment_max_freq_shift:      max freq shift (fraction of total frequency range)
        augment_max_contrast:        max random contrast factor
        shuffle:                     whether to shuffle training paths at each epoch end
    '''
    
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(lambda x: _parse_tfrecord(x, image_shape))

    if augment:
        if augment_max_contrast>1:
            ds = ds.map(lambda x, y: (random_contrast(x, augment_max_contrast), y))
            
        if augment_max_time_masks>0:
            ds = ds.map(lambda x, y: (time_mask(x,
                                               augment_max_time_masks,
                                               augment_max_time_mask_size), y))
        if augment_max_freq_masks>0:
            ds = ds.map(lambda x, y: (freq_mask(x,
                                               augment_max_freq_masks,
                                               augment_max_freq_mask_size), y))

        ds = ds.map(lambda x, y: (affine_transform(x,
                                                  augment_max_time_shift,
                                                  augment_max_freq_shift), y))
                        
    return ds.batch(batch_size)
    
        
def _parse_tfrecord(example_proto, shape):
    feature = {
        'X': tf.io.FixedLenFeature(shape[0]*shape[1], tf.float32),
        'Y': tf.io.FixedLenFeature([], tf.float32)
    }
    features = tf.io.parse_example([example_proto], features=feature)
    features['X'] = tf.reshape(features['X'], shape)
    return features['X'], features['Y']
                    
        
def time_mask(x, maxmasks, maxwidth):
    
    limit = tf.shape(x)[1]
    for i in range(tf.random.uniform([], 0, maxmasks)):
        f = tf.random.uniform(shape=(), minval=0, maxval=int(tf.cast(limit, tf.float32)*maxwidth), dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=limit - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(limit), (1, -1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
        )
        x = tf.where(condition, tf.cast(tf.reduce_mean(x), x.dtype), x)
    return x


def freq_mask(x, maxmasks, maxwidth):
    
    limit = tf.shape(x)[0]
    for i in range(tf.random.uniform([], 0, maxmasks)):
        f = tf.random.uniform(shape=(), minval=0, maxval=int(tf.cast(limit, tf.float32)*maxwidth), dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=limit - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(limit), (-1, 1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
        )
        x = tf.where(condition, tf.cast(tf.reduce_mean(x), x.dtype), x)
    return x


def affine_transform(x: tf.Tensor, time_shift_percent: float=0.0, freq_shift_percent: float=0.0):
    ''' Apply affine time/frequency shifting to a spectrogram
    
    Args:
        x: tensor sample
        time_shift_percent: fraction of total time indicating max possible time shift
        freq_shift_percent: fraction of total freq range indicating max possible freq shift
    '''
    orig_shape = x.shape
    if time_shift_percent>0:
        time_pad = tf.random.uniform([], 0, int(x.shape[1]*time_shift_percent), tf.int32)
    else:
        time_pad = 0
    if freq_shift_percent>0:
        freq_pad = tf.random.uniform([], 0, int(x.shape[0]*freq_shift_percent), tf.int32)
    else:
        freq_pad = 0
    x = tf.pad(x, [[freq_pad, freq_pad], [time_pad, time_pad]], mode='CONSTANT', constant_values=tf.reduce_mean(x))
    if time_pad>0:
        time_shift = tf.random.uniform([], 0, int(time_pad*2), tf.int32)
    else:
        time_shift = 0
    if freq_pad>0:
        freq_shift = tf.random.uniform([], 0, int(freq_pad*2), tf.int32)
    else:
        freq_shift = 0
    x = x[freq_shift:(freq_shift+orig_shape[0]), time_shift:(time_shift+orig_shape[1])] 
    return x
    
    
def random_contrast(x, maxval):
    x = tf.image.random_contrast(tf.expand_dims(x, axis=0), 1, maxval)[0]
    return x


def get_files_and_label_map(data_dirs: list, 
                            train_split: float=1.0, 
                            classes: Optional[list]=None, 
                            target_train: Optional[int]=None, 
                            target_val: Optional[int]=None, 
                            ext: str='_spec.npy'):
    ''' Prepares train/val path lists and a label dictionary

    Args:
        data_dirs:     list of paths to directories in which to search for files
        train_split:   portion of data per class to use for training
        classes:       list of classes to get files for
        target_train:  desired number of training samples per class for resampling
        target_val:    desired number of validation samples per class for resampling
        ext:           suffix of files to collect
    Returns:
        files_train:   list of training file paths
        files_val:     list of validation file paths
        labels:        class dictionary
    '''
    if not isinstance(data_dirs, list): # check arg is list
        data_dirs = [data_dirs]
    
    for i in range(len(data_dirs)):
        if not data_dirs[i].endswith('/'):
            data_dirs[i]+='/'
        
    if classes is None: # maybe get class list
        classes = set()
        for i in data_dirs:
            for j in sorted(os.listdir(i)):
                if j.startswith('.'):
                    continue
                if os.path.isdir(i+'/'+j):
                    classes.add(j)
        classes = sorted(list(classes))
            
    files_train = list()
    files_val = list()
    label_map = dict()
    if ('background' in classes):
        classes = ['background']+[i for i in classes if i!='background']
    for cnt, i in enumerate(classes): # loop over classes
        label_map[i]=cnt # add class to dict
        
        # get class files
        class_train = []
        class_val = []
        for dr in data_dirs:
            if not os.path.exists(dr+i):
                continue
            tmp = [dr+i+'/'+j for j in sorted(os.listdir(dr+i)) if j.endswith(ext)]
            # split data
            if train_split==1.0:
                tmp_train=tmp
                tmp_val=[]
            else:
                tmp_train, tmp_val = train_test_split(tmp,
                                                      train_size=train_split)
            class_train += tmp_train
            class_val += tmp_val
            
        # maybe resample
        if target_train is not None:
            class_train = resample_files(class_train, target_train)
        if target_val is not None:
            class_val = resample_files(class_val, target_val)
            
        files_train += class_train
        files_val += class_val
                
    return files_train, files_val, label_map


def resample_files(x: list, target: int) -> list:
    ''' Resamples a list
    '''
    if len(x)>target:
        x = list(np.random.choice(x, target, replace=0)) # downsample
    elif len(x)<target:
        x = x+list(np.random.choice(x, target-len(x), replace=1)) # upsample without losing information
    return x
            

def num_pos_neg_per_class(files: list):
    ''' Display the number of positive and negative samples per class
    
        Expects file path format:

            .../<'positive' or 'negative'>/<class_name>/<file_name>
    Args:
        files: list of file paths
    '''
    classes = []
    p = []
    n = []
    for i in list(set([i.split('/')[-2] for i in files])):
        classes.append(i)
        p.append(len(np.where([(j.split('/')[-2]==i) & (j.split('/')[-3]=='positive') for j in files])[0]))
        n.append(len(np.where([(j.split('/')[-2]==i) & (j.split('/')[-3]=='negative') for j in files])[0]))
    display(pd.DataFrame({'Class':classes,'Positives':p,'Negatives':n}))
    
    
def plot_batch_samples(batch: tf.Tensor, nr=4, nc=4):
    ''' Plots a batch of examples from the DataGenerator
    
    Args:
        batch: batch tensor
        nr:    number of rows to plot
        nc:    number of columns to plot
    '''
    plt.figure(figsize=(15,15))
    for c in range(len(batch)):
        plt.subplot(nr,nc,c+1)
        plt.pcolormesh(batch[c].numpy())
        plt.clim([-100, 20])
        plt.axis('off') 
        
        
def create_tfrecords(files: list,
                     labels: np.ndarray,
                     outdir: str,
                     batch_size: int = 1000):
    ''' Creates tfrecord files for batches of data
    
    Args:
        files:    list of spectrogram files to store in tfrecords
        labels:   label array with first dimension equal to the number of files
        outdir:   directory to store the tfrecord files
    '''
    if not outdir.endswith('/'):
        outdir+='/'
    for i in range(0, len(files), batch_size):
        batch = []
        batch_labels = []
        for j in range(batch_size):
            if (i+j)>=len(files):
                continue
            sample = np.load(files[i+j])
            sample = sample.flatten()[np.newaxis,...]
            batch.append(sample)
            batch_labels.append(labels[i+j])
        batch = np.vstack(batch)
        batch_labels = np.vstack(batch_labels)
        np_to_tfrecords(batch, 
                        batch_labels, 
                        outdir+'files_'+str(i)+'-'+str(np.min([len(files), i+batch_size]))+'.tfrecords')
        
                     
def np_to_tfrecords(X, Y, file_path_prefix):
    '''
    Adapted from https://gist.github.com/Geoyi/0b4f304143b7480b2897f94cf3587a67
    
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    
    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.
    
    '''
    def _dtype_feature(ndarray):
        '''match appropriate tf.train.Feature class with dtype of ndarray. '''
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))
            
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank, 
                               # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)            
    
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrec'
    writer = tf.io.TFRecordWriter(result_tf_file)
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]
        
        d_feature = {}
        d_feature['X'] = dtype_feature_x(x)
        if Y is not None:
            d_feature['Y'] = dtype_feature_y(y)
            
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
        


    
    