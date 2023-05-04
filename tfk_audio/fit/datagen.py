import os
from typing import Optional, Callable
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
from .labels import *

AUTO = tf.data.AUTOTUNE


def spectrogram_dataset_from_tfrecords(files: list,
                                       image_shape: tuple,
                                       nclass: int,
                                       batch_size: int = 1,
                                       time_crop: int = None,
                                       random_time_crop: bool = False,
                                       augment: bool = False,
                                       augment_blend_prob: bool = 0.5,
                                       augment_blend_strength: float = 0.5,
                                       augment_mixup: bool = False,
                                       augment_max_time_masks: int = 5,
                                       augment_max_time_mask_size: float = 0.1,
                                       augment_max_freq_masks: int = 5,
                                       augment_max_freq_mask_size: float = 0.1,
                                       augment_add_noise_prob: float = 0.0,
                                       augment_add_noise_stds: float = 0.5,
                                       augment_max_time_shift: float = 0.25,
                                       augment_max_freq_shift: float = 0.05,
                                       augment_max_contrast: float = 2.0,
                                       assume_negative_prob: float = 0.0,
                                       label_weights: tuple = (1, 1),
                                       repeat = False):
    ''' Prepares a tf.data.Dataset for generating spectrogram training data
    
    Args:
        files:                       list of paths to tfrecord files
        image_shape:                 integer tuple of spectrogram image shape: (frequency bins, time bins)
        batch_size:                  samples per batch
        time_crop:                   None or int indicating width to crop spectrograms to
        random_time_crop:            boolean indicating whether time crops should be randomly shifted
        augment:                     whether to apply data augmentation
        augment_blend_prob:          probability of blending a sample with another
        augment_blend_strength:      strength of blended samples
        augment_mixup:               whether to apply mixup augmentation; not recommended along with blending
        augment_max_time_masks:      max number of masks to apply in the time axis (https://arxiv.org/abs/1904.08779)
        augment_max_time_mask_size:  max desired mask width (fraction of total time)
        augment_max_freq_masks:      max number of masks to apply in the freq axis (https://arxiv.org/abs/1904.08779)
        augment_max_freq_mask_size:  max desired mask width (fraction of total frequency range)
        augment_add_noise_prob:      probability of adding noise to a sample
        augment_add_noise_stds:      standard deviation of added Gaussian noise (# of batch standard deviations)
        augment_max_time_shift:      max time shift (fraction of total time)
        augment_max_freq_shift:      max freq shift (fraction of total frequency range)
        augment_max_contrast:        max random contrast factor
        assume_negative_prob:        probability of randomly setting a -1 label to 0
        label_weights:               a tuple containing the weight of negative and positive samples, respectively
        repeat:                      whether to repeat the data infinitely
    '''    
    ds = tf.data.Dataset.from_tensor_slices(files) # list of tfrecord files
    ds = ds.shuffle(len(files), reshuffle_each_iteration=True) # shuffle tfrecord files
    if repeat:
        ds = ds.repeat()
    ds = ds.interleave(tf.data.TFRecordDataset, 
                       cycle_length=max(1, len(files)//100), 
                       block_length=1) # load tfrecords
    ds = ds.map(lambda x: _parse_tfrecord(x, image_shape, nclass), 
                num_parallel_calls=AUTO) # parse records
    ds = ds.shuffle(batch_size*2, 
                    reshuffle_each_iteration=True) # use buffer shuffling
        
    if time_crop<image_shape[0]:
        # crop the sample in time
        # doing this turns the results into a tensor with undefined time width, so some of the following operations
        #     require the original image shape to be passed
        ds = ds.map(lambda x, y: (_time_crop(x, time_crop, image_shape[0], random_time_crop), y), 
                    num_parallel_calls=AUTO)
        
    if augment:
        
        # random contrast
        if augment_max_contrast>1:
            ds = ds.map(lambda x, y: (random_contrast(x, augment_max_contrast), y), 
                        num_parallel_calls=AUTO)
        
        # blending
        if augment_blend_prob>0:
            ds = ds.batch(batch_size)
            ds = ds.map(lambda x, y: blend(x, y, batch_size, augment_blend_prob, augment_blend_strength), 
                        num_parallel_calls=AUTO)
            ds = ds.unbatch()

            
        # mixup
        if augment_mixup:
            ds = ds.batch(batch_size)
            ds = ds.map(lambda x, y: mixup(x, y, batch_size), 
                        num_parallel_calls=AUTO)
            ds = ds.unbatch()
            
        # add noise
        if augment_add_noise_prob>0:
            ds = ds.map(lambda x, y: (add_noise(x, 
                                                augment_add_noise_prob,
                                                augment_add_noise_stds), y), 
                        num_parallel_calls=AUTO)
            
        # time masks
        if augment_max_time_masks>0:
            ds = ds.map(lambda x, y: (time_mask(x,
                                                augment_max_time_masks,
                                                augment_max_time_mask_size), y), 
                        num_parallel_calls=AUTO)
        # freq masks 
        if augment_max_freq_masks>0:
            ds = ds.map(lambda x, y: (freq_mask(x,
                                                augment_max_freq_masks,
                                                augment_max_freq_mask_size), y), 
                        num_parallel_calls=AUTO)
        # affine time-freq shift
        if (augment_max_time_shift>0) or (augment_max_freq_shift>0):
            ds = ds.map(lambda x, y: (affine_transform(x,
                                                       (time_crop, image_shape[1]),
                                                       augment_max_time_shift,
                                                       augment_max_freq_shift), y), 
                        num_parallel_calls=AUTO)
    
    if assume_negative_prob>0:
        ds = ds.map(lambda x, y: (x, assume_negative(y, (nclass,), assume_negative_prob)),
                    num_parallel_calls=AUTO)
        
    
    # add sample weight
    if label_weights!=(1, 1):
        ds = ds.map(lambda x, y: [x, y, tf.gather(label_weights, tf.cast(tf.reduce_max(y), tf.int32))], 
                    num_parallel_calls=AUTO)

    return ds.batch(batch_size).shuffle(5, reshuffle_each_iteration=True) # batch and shuffle batches
    
    
def assume_negative(y, shape, prob=0.25):
    ''' Sets -1 labelst to 0 with some probability
    '''
    assumptions = tf.where(tf.random.uniform(shape, 0, 1)<=prob, tf.zeros_like(y), tf.ones_like(y)*-1)
    return tf.where(y==-1, tf.cast(assumptions, tf.float32), y)
    
    
def _parse_tfrecord(example_proto, shape, nclass):
    feature = {
        'X': tf.io.FixedLenFeature(shape[0]*shape[1], tf.float32),
        'Y': tf.io.FixedLenFeature(nclass, tf.float32)
    }
    features = tf.io.parse_example([example_proto], features=feature)
    features['X'] = tf.reshape(features['X'], shape)
    features['Y'] = tf.squeeze(features['Y'])
    return features['X'], features['Y']
       
    
def _time_crop(x, width, image_width, random):
    if not random:
        start = (image_width-width)//2
        return x[..., start: (start+width), :]
    else:
        start = tf.random.uniform([], 0, image_width-width, dtype=tf.int32)
        return x[..., start: (start+width), :]
    
        
def freq_mask(x, maxmasks, maxwidth):
    ''' Apply time masks to a spectrogram
    '''
    limit = tf.shape(x)[1]
    for _ in range(tf.random.uniform([], 0, maxmasks+1, dtype=tf.int32)):
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


def time_mask(x, maxmasks, maxwidth):
    ''' Apply frequency masks to a spectrogram
    '''
    limit = tf.shape(x)[0]
    for _ in range(tf.random.uniform([], 0, maxmasks+1, dtype=tf.int32)):
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


def affine_transform(x: tf.Tensor, image_shape: tuple, time_shift_percent: float=0.0, freq_shift_percent: float=0.0):
    ''' Apply affine time/frequency shifting to a spectrogram
    
    Args:
        x: tensor sample
        time_shift_percent: fraction of total time indicating max possible time shift
        freq_shift_percent: fraction of total freq range indicating max possible freq shift
    '''
    if time_shift_percent>0:
        time_pad = tf.random.uniform([], 0, int(image_shape[0]*time_shift_percent), tf.int32)
    else:
        time_pad = 0
    if freq_shift_percent>0:
        freq_pad = tf.random.uniform([], 0, int(image_shape[1]*freq_shift_percent), tf.int32)
    else:
        freq_pad = 0
    x = tf.pad(x, [[time_pad, time_pad], [freq_pad, freq_pad]], mode='CONSTANT', constant_values=tf.reduce_mean(x))
    if time_pad>0:
        time_shift = tf.random.uniform([], 0, int(time_pad*2), tf.int32)
    else:
        time_shift = 0
    if freq_pad>0:
        freq_shift = tf.random.uniform([], 0, int(freq_pad*2), tf.int32)
    else:
        freq_shift = 0
    x = x[time_shift:(time_shift+image_shape[0]), freq_shift:(freq_shift+image_shape[1])] 
    return x
    
    
def random_contrast(x, maxval):
    ''' Randomly adjust image contrast
    '''
    x = tf.image.random_contrast(tf.expand_dims(x, axis=0), 1, maxval)[0]
    return x


def add_noise(x, prob, strength):
    ''' Add Gaussian noise
    '''
    if tf.random.uniform([],0,1)<prob:
        x = x + tf.clip_by_value(
            tf.random.normal(tf.shape(x), 0, strength*tf.math.reduce_std(x)),
            -strength*tf.math.reduce_std(x),
            strength*tf.math.reduce_std(x)
        )
    return x


def mixup(X: tf.Tensor, 
          y: tf.Tensor,
          batch_size: int):
    ''' Apply mix up augmentation to a batch of spectrograms
    
    Args:
        X: batch input tensor
        y: batch label tensor
    Returns:
        X: blended batch input tensor
        y: blended batch label tensor
    '''
    tf.debugging.assert_non_negative(tf.reduce_min(y), 
                                     message='Error: mixup augmentation not yet compatible with unknown (-1) labels')
    
    toblend = tf.reshape(beta_dist(batch_size), (batch_size,1,1))
    idxshuffle = tf.random.shuffle(tf.range(batch_size))
    
    X = X*(1-toblend) + tf.gather(X, idxshuffle, axis=0)*toblend
    y = y*(1-toblend) + tf.gather(y, idxshuffle, axis=0)*toblend
    
    return X, y


def blend(X: tf.Tensor, 
          y: tf.Tensor,
          batch_size: int,
          prob: float=1.0, 
          strength: float=0.5):
    ''' Apply blending augmentation to a batch of spectrograms
    
    Args:
        X: batch input tensor
        y: batch label tensor
    Returns:
        X: blended batch input tensor
        y: blended batch label tensor
    '''
    # binary vector of length batch_size indicating whether the sample will be blended
    toblend = tf.where(tf.math.logical_and(tf.random.uniform((batch_size, 1, 1), 0, 1)<=prob, 
                                           tf.reshape(tf.reduce_max(y, axis=1), (-1, 1, 1))==1), # only blend positive samples
                       tf.ones_like((batch_size,)), 
                       tf.zeros_like((batch_size,)))
    blendv = tf.cast(toblend, tf.float32) * strength
    idxshuffle = tf.random.shuffle(tf.range(batch_size))
    X = X*(1-blendv) + tf.gather(X, idxshuffle, axis=0)*blendv

    # combine labels
    labels = combine_labels(y, tf.gather(y, idxshuffle, axis=0), tf.cast(tf.reshape(toblend, [-1]), tf.bool))

    return X, y


def combine_labels(y1: tf.Tensor, y2: tf.Tensor, idx: tf.Tensor):
    ''' Implements logic for combining multi-label vectors with compatibility for unknown labels represented by -1
    '''
    y1t = tf.where(y1==1, tf.ones_like(y1)*2, y1)
    y2t = tf.where(y2==1, tf.ones_like(y2)*2, y2)
    y1 = tf.where(tf.tile(tf.expand_dims(idx, axis=1), multiples=[1, y1.shape[1]]), y1t+y2t, y1)
    return tf.clip_by_value(y1, -1, 1)
    

def beta_dist(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def get_files_and_label_map(data_dirs: list, 
                            train_split: float=1.0, 
                            classes: Optional[list]=None, 
                            target_train: Optional[int]=None, 
                            target_val: Optional[int]=None, 
                            ext: str='_spec.npy',
                            random_state: Optional[int]=None):
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
                                                      train_size=train_split,
                                                      random_state=random_state)
            class_train += tmp_train
            class_val += tmp_val
            
        # maybe resample
        if target_train is not None:
            class_train = resample_files(class_train, target_train, random_state)
        if target_val is not None:
            class_val = resample_files(class_val, target_val, random_state)
            
        files_train += class_train
        files_val += class_val
                
    return files_train, files_val, label_map


def resample_files(x: list, target: int, rand_state: int) -> list:
    ''' Resamples a list
    '''
    if len(x)>target:
        x = list(resample(x, n_samples=target, replace=0, random_state=rand_state)) # downsample
    elif len(x)<target:
        x = x+list(resample(x, n_samples=target-len(x), replace=1, random_state=rand_state)) # upsample without losing information
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
    df = pd.DataFrame({'Class':classes,'Positives':p,'Negatives':n})
    return df
    
    
def plot_batch_samples(batch: tf.Tensor, nr=4, nc=4, dblims=(-100, 20)):
    ''' Plots a batch of examples from the DataGenerator
    
    Args:
        batch: batch tensor
        nr:    number of rows to plot
        nc:    number of columns to plot
    '''
    plt.figure(figsize=(15,15))
    for c in range(nr*nc):
        plt.subplot(nr,nc,c+1)
        plt.pcolormesh(batch[c].numpy().T)
        plt.clim(dblims)
        plt.axis('off') 
        
        
def create_tfrecords(files: list,
                     labels: np.ndarray,
                     outdir: str,
                     batch_size: int = 1000,
                     overwrite: bool = True,
                     update: int = 1000):
    ''' Creates tfrecord files for batches of data
    
    Args:
        files:      list of spectrogram files to store in tfrecords
        labels:     label array with first dimension equal to the number of files
        outdir:     directory to store the tfrecord files
        batch_size: max number of samples per tfrecord file
        
    Returns:
        tfrecord_files: list of tfrecord file paths
    '''
    tfrecord_files = []
    if not outdir.endswith('/'):
        outdir+='/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in os.listdir(outdir):
        if i.endswith('.tfrec'):
            if overwrite:
                os.remove(outdir+i)
    for i in range(0, len(files), batch_size):
        if i%update==0:
            print(i)
        if not overwrite:
            if os.path.exists(outdir+'files_'+str(i)+'-'+str(np.min([len(files), i+batch_size]))+'.tfrec'):
                tfrecord_files.append(outdir+'files_'+str(i)+'-'+str(np.min([len(files), i+batch_size]))+'.tfrec')
                continue
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
                        outdir+'files_'+str(i)+'-'+str(np.min([len(files), i+batch_size])))
        tfrecord_files.append(outdir+'files_'+str(i)+'-'+str(np.min([len(files), i+batch_size]))+'.tfrec')
        
    return tfrecord_files

                     
def np_to_tfrecords(X, Y, file_path_prefix):
    '''
    Adapted from https://gist.github.com/Geoyi/0b4f304143b7480b2897f94cf3587a67
    
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.
    
    Args:
        X : numpy.ndarray of rank 2
            Numpy array for training inputs. Its dtype should be float32, float64, or int64.
            If X has a higher rank, it should be rshape before fed to this function.
        Y : numpy.ndarray of rank 2 or None
            Numpy array for training labels. Its dtype should be float32, float64, or int64.
            None if there is no label array.
        file_path_prefix : str
            The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    
    Raises:
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
            raise ValueError('The input should be numpy ndarray. \
                               Instaed got {}'.format(ndarray.dtype))
            
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
    
    # Create tfrecord writer
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
    
    writer.close()
        


    
    
