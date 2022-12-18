import os
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




def get_files_and_label_map(data_dirs: list, 
                            train_split: float=1.0, 
                            classes: Optional[list]=None, 
                            target_train: Optional[int]=None, 
                            target_val: Optional[int]=None, 
                            ext: str='_spec.npy'):
    ''' Prepares train/val path lists and a label dictionary

    Args:
        data_dirs: list of paths to directories in which to search for files
        train_split: portion of data per class to use for training
        classes: list of classes to get files for
        target_train: desired number of training samples per class for resampling
        target_val: desired number of validation samples per class for resampling
        ext: suffix of files to collect
    Returns:
        files_train: list of training file paths
        files_val: list of validation file paths
        labels: class dictionary
    '''
    if not isinstance(data_dirs, list): # check arg is list
        data_dirs = [data_dirs]
        
    if classes is None: # maybe get class list
        classes = set()
        for i in data_dirs:
            for j in os.listdir(i):
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
            tmp = [dr+i+'/'+j for j in os.listdir(dr+i) if j.endswith(ext)]
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


def get_labels_pos_neg_per_class(files: list, label_map: dict) -> np.ndarray:
    ''' Create a label array from the given path list
    
        Expects file path format:

            .../<'positive' or 'negative'>/<class_name>/<file_name>

    Args:
        files: list of file paths
        label_map: dictionary mapping each class in the file paths to an index
    Returns:
        y: label array; unknown values are indicated by -1
    '''
    y = np.ones((len(files),len(label_map)))*-1
    for c,i in enumerate(files):
        assert i.split('/')[-3] in ('positive', 'negative'), 'Error: Label could not be interpretted.'
        if i.split('/')[-3]=='positive':
            y[c, int(label_map[i.split('/')[-2]])] = 1
        else:
            y[c, int(label_map[i.split('/')[-2]])] = 0
    return y 
            

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
        nr: number of rows to plot
        nc: number of columns to plot
    '''
    plt.figure(figsize=(15,15))
    for c in range(len(batch)):
        plt.subplot(nr,nc,c+1)
        plt.pcolormesh(batch[c].numpy())
        plt.clim([-100, 20])
        plt.axis('off')
    
    
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
                       shuffle: bool = True,
                       assume_absent: bool = True,
                       label_format: str = 'multi-label'):
    ''' Prepares a tf.data.Dataset for generating spectrogram training data
    
    Args:
        files:                       list of paths to tfrecord files
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
        assume_absent:               if True, will convert any -1 (unknown) labels to 0 (absent)
        label_format:                'multi-label', 'single-label', or 'multi-class'   
    '''
    
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(lambda x: _parse_tfrecord(x, image_shape))

    if augment:
        if augment_max_contrast>1:
            ds = ds.map(lambda x: random_contrast(x, augment_max_contrast))
            
        if augment_max_time_masks>0:
            ds = ds.map(lambda x: time_mask(x, 
                                            augment_max_time_masks, 
                                            augment_max_time_mask_size))
        if augment_max_freq_masks>0:
            ds = ds.map(lambda x: freq_mask(x, 
                                            augment_max_freq_masks, 
                                            augment_max_freq_mask_size))

        ds = ds.map(lambda x: affine_transform(x, 
                                               augment_max_time_shift, 
                                               augment_max_freq_shift))
                        
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
    
    limit = tf.shape(x[0])[1]
    for i in range(tf.random.uniform([], 0, maxmasks)):
        f = tf.random.uniform(shape=(), minval=0, maxval=int(tf.cast(limit, tf.float32)*maxwidth), dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=limit - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(limit), (1, -1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
        )
        x[0] = tf.where(condition, tf.cast(tf.reduce_mean(x[0]), x[0].dtype), x[0])
    return x


def freq_mask(x, maxmasks, maxwidth):
    
    limit = tf.shape(x[0])[0]
    for i in range(tf.random.uniform([], 0, maxmasks)):
        f = tf.random.uniform(shape=(), minval=0, maxval=int(tf.cast(limit, tf.float32)*maxwidth), dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=limit - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(limit), (-1, 1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
        )
        x[0] = tf.where(condition, tf.cast(tf.reduce_mean(x[0]), x[0].dtype), x[0])
    return x


def affine_transform(x: tf.Tensor, time_shift_percent: float=0.0, freq_shift_percent: float=0.0):
    ''' Apply affine time/frequency shifting to a spectrogram
    
    Args:
        x: tensor sample
        time_shift_percent: fraction of total time indicating max possible time shift
        freq_shift_percent: fraction of total freq range indicating max possible freq shift
    '''
    orig_shape = x[0].shape
    if time_shift_percent>0:
        time_pad = tf.random.uniform([], 0, int(x[0].shape[1]*time_shift_percent), tf.int32)
    else:
        time_pad = 0
    if freq_shift_percent>0:
        freq_pad = tf.random.uniform([], 0, int(x[0].shape[0]*freq_shift_percent), tf.int32)
    else:
        freq_pad = 0
    x[0] = tf.pad(x[0], [[freq_pad, freq_pad], [time_pad, time_pad]], mode='CONSTANT', constant_values=tf.reduce_mean(x[0]))
    if time_pad>0:
        time_shift = tf.random.uniform([], 0, int(time_pad*2), tf.int32)
    else:
        time_shift = 0
    if freq_pad>0:
        freq_shift = tf.random.uniform([], 0, int(freq_pad*2), tf.int32)
    else:
        freq_shift = 0
    x[0] = x[0][freq_shift:(freq_shift+orig_shape[0]), time_shift:(time_shift+orig_shape[1])] 
    return x
    
    
def random_contrast(x, maxval):
    
    x[0] = tf.image.random_contrast(tf.expand_dims(x[0], axis=0), 1, maxval)[0]
    return x    


    
    