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
                                       augment_blend_req_pos: bool = False,
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
                                       label_smoothing: float = 0.0,
                                       label_weights: tuple = (1, 1),
                                       repeat = False):
    ''' Prepares a tf.data.Dataset for generating spectrogram training data
    
    Args:
        files:                       list of paths to tfrecord files
        image_shape:                 integer tuple of spectrogram image shape: (frequency bins, time bins)
        nclass:                      number of classes for parsing
        batch_size:                  samples per batch
        time_crop:                   None or int indicating width to crop spectrograms to
        random_time_crop:            boolean indicating whether time crops should be randomly shifted
        augment:                     whether to apply data augmentation
        augment_blend_prob:          probability of blending a sample with another
        augment_blend_strength:      strength of blended samples
        augment_blend_req_pos:       whether to require at least one positive sample for blending
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
        label_smoothing:             amount of label smoothing to apply. 
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

    if time_crop is None:
        time_crop = image_shape[0]
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
            ds = ds.map(lambda x, y: blend(x, y, batch_size, augment_blend_prob, augment_blend_req_pos, augment_blend_strength), 
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
    
    if label_smoothing:
        # Apply label smoothing to the labels
        ds = ds.map(lambda x, y: (x, _label_smoothing(y, label_smoothing)), num_parallel_calls=AUTO)
    
    # add sample weight
    if label_weights!=(1, 1):
        label_weights = (0, *label_weights)
        ds = ds.map(lambda x, y: [x, y, tf.gather(label_weights, tf.cast(tf.reduce_max(y)+1, tf.int32))], 
                    num_parallel_calls=AUTO)

    return ds.batch(batch_size).shuffle(5, reshuffle_each_iteration=True) # batch and shuffle batches
    
    
def _label_smoothing(y: tf.Tensor, alpha: float = 0.1) -> tf.Tensor:
    """
    Apply label smoothing to the input tensor.

    Args:
        y (tf.Tensor): The input tensor with integer labels.
        alpha (float): The smoothing parameter. Default is 0.1.

    Returns:
        tf.Tensor: The tensor with smoothed labels.

    Notes:
        This function performs label smoothing by subtracting alpha from positive labels (> 0)
        and assigning alpha / number of classes to labels equal to zero.

    Example:
        >>> y = tf.constant([[0, -1, 1], [0, 1, 0], [1, 0, 0]])
        >>> y_smoothed = label_smoothing(y, alpha=0.1)
        >>> print(y_smoothed.numpy())
    """
    # Smooth only the positive values (greater than 0)
    y_smoothed = tf.where(y > 0, y - alpha, y)

    # For values equal to zero, assign alpha / number of classes
    num_classes = tf.shape(y)[-1]
    y_smoothed = tf.where(y == 0, alpha / tf.cast(num_classes, dtype=tf.float32), y_smoothed)

    return y_smoothed

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


def blend(X: tf.Tensor, 
          y: tf.Tensor,
          batch_size: int,
          prob: float=1.0,
          augment_blend_req_pos=False, 
          strength: float=0.5):
    ''' Apply blending augmentation to a batch of spectrograms
    
    Args:
        X: batch input tensor
        y: batch label tensor
    Returns:
        X: blended batch input tensor
        y: blended batch label tensor
    '''
    # boolean vector of length batch_size indicating whether the sample will be blended
    # according to user defined probability
    toblend_prob = tf.random.uniform((batch_size, 1, 1), 0, 1) <= prob

    if augment_blend_req_pos:
        # add 2nd condition where at least one positive will be present in the final blended sample
        # boolean vector of length batch_size indicating whether each sample will be blended based on if has at least 1
        # positive in it
        toblend_positive = tf.reshape(tf.reduce_max(y, axis=1), (-1, 1, 1)) == 1
        
        # length batch_size indicating whether the sample will be blended based on blend_prob and 
        # if the sample in the vector is a positive
        toblend = tf.where(tf.logical_and(toblend_prob, toblend_positive), 1, 0)
                                           
    else:
        # length batch_size indicating whether the sample will be blended based solely on blend_prob
        toblend = tf.where(toblend_prob, 
                       tf.ones_like((batch_size,)), 
                       tf.zeros_like((batch_size,)))

    # Multiplies, to_blend by the strength, 0s remain zeros, 1s turn into strength 
    blendv = tf.cast(toblend, tf.float32) * strength
    
    # Shuffle the indices of each sample in the vector, this will be used to
    # randomly select a second sample to blend
    # results in vector of length batch_size with different indices
    idxshuffle = tf.random.shuffle(tf.range(batch_size))

    # For each sample selected to be blended, select a second sample from the batch
    # while applying the strenght of each
    X = X*(1-blendv) + tf.gather(X, idxshuffle, axis=0)*blendv

    # combine labels
    labels = combine_labels(y, tf.gather(y, idxshuffle, axis=0), tf.cast(tf.reshape(toblend, [-1]), tf.bool))

    return X, labels


def combine_labels(y1: tf.Tensor, y2: tf.Tensor, idx: tf.Tensor):
    ''' Implements logic for combining multi-label vectors with compatibility for unknown labels represented by -1
    '''
    y1t = tf.where(y1==1, tf.ones_like(y1)*2, y1)
    y2t = tf.where(y2==1, tf.ones_like(y2)*2, y2)
    y1 = tf.where(tf.tile(tf.expand_dims(idx, axis=1), multiples=[1, y1.shape[1]]), y1t+y2t, y1)
    return tf.clip_by_value(y1, -1, 1)


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
    tf.debugging.assert_non_negative(
        tf.reduce_min(y),
        message='Error: mixup augmentation not yet compatible with unknown (-1) labels'
    )
    
    toblend = tf.reshape(beta_dist(batch_size), (batch_size,1,1))
    idxshuffle = tf.random.shuffle(tf.range(batch_size))
    
    X = X*(1-toblend) + tf.gather(X, idxshuffle, axis=0)*toblend
    y = y*(1-toblend) + tf.gather(y, idxshuffle, axis=0)*toblend
    
    return X, y


def beta_dist(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def get_files_and_label_map(data_dirs: list, 
                            train_split: float=1.0, 
                            test_split: float=0.0,
                            classes: Optional[list]=None, 
                            target_train: Optional[int]=None, 
                            target_val: Optional[int]=None, 
                            target_test: Optional[int]=None, 
                            ext: str='_spec.npy',
                            random_state: Optional[int]=None):
    ''' Prepares train/val/test path lists and a label dictionary

    Args:
        data_dirs:     list of paths to directories in which to search for files
        train_split:   portion of data per class to use for training
        test_split:    portion of validation data per class to use for testing
        classes:       list of classes to get files for
        target_train:  desired number of training samples per class for resampling
        target_val:    desired number of validation samples per class for resampling
        target_test:   desired number of testing samples per class for resampling
        ext:           suffix of files to collect
        
    Returns:
        files_train:   list of training file paths
        files_val:     list of validation file paths
        files_test:    list of testing file paths
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
            
    files_train, files_val, files_test = [], [], []
    label_map = dict()
    if ('background' in classes):
        classes = ['background']+[i for i in classes if i!='background'] # put background class first
    for cnt, i in enumerate(classes):  # loop over classes
        label_map[i]=cnt  # add class to dict
        
        # get class files
        class_train, class_val, class_test = [], [], []
        for dr in data_dirs:
            if not os.path.exists(dr+i):
                continue
            tmp = [dr+i+'/'+j for j in sorted(os.listdir(dr+i)) if j.endswith(ext)]
            # split data
            if train_split == 1.0:
                tmp_train = tmp
                tmp_val ,tmp_test = [], []
            else:
                tmp_train, tmp_val = train_test_split(tmp,
                                                      train_size=train_split,
                                                      random_state=random_state)
                tmp_test = []
                
                if test_split:
                    tmp_val, tmp_test = train_test_split(tmp_val,
                                                      test_size=test_split,
                                                      random_state=random_state)
            class_train += tmp_train
            class_val += tmp_val
            class_test += tmp_test
            
        # maybe resample
        if target_train is not None:
            class_train = resample_files(class_train, target_train, random_state)
        if target_val is not None:
            class_val = resample_files(class_val, target_val, random_state)
        if target_test is not None:
            class_test = resample_files(class_test, target_test, random_state)
            
        files_train += class_train
        files_val += class_val
        files_test += class_test
                
    return files_train, files_val, files_test, label_map


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
        if c<batch.shape[0]:
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
    print('Creating tfrecord files in '+outdir)
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
    Converts NumPy arrays into a TFRecord file for use in TensorFlow.

    Args:
        X : np.ndarray of shape (n_samples, n_features)
            Input features. Must be of type float32, float64, or int64.
        Y : np.ndarray of shape (n_samples, n_labels) or None
            Labels corresponding to X. Must match the number of samples in X if provided.
        file_path_prefix : str
            Prefix for the output TFRecord file (without the '.tfrec' extension).

    Raises:
        ValueError: If input data types are unsupported.
    '''
    
    def _get_feature_fn(dtype):
        if dtype in [np.float32, np.float64]:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array.")
    
    if Y is not None:
        if not isinstance(Y, np.ndarray) or Y.ndim != 2:
            raise ValueError("Y must be a 2D NumPy array or None.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
    
    feature_fn_x = _get_feature_fn(X.dtype)
    feature_fn_y = _get_feature_fn(Y.dtype) if Y is not None else None

    output_path = file_path_prefix + '.tfrec'
    with tf.io.TFRecordWriter(output_path) as writer:
        for i in range(X.shape[0]):
            features = {
                'X': feature_fn_x(X[i])
            }
            if Y is not None:
                features['Y'] = feature_fn_y(Y[i])
            
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())

        


    
    
