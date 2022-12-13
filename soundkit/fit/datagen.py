import os
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

def get_files_and_label_map(data_dirs: list, 
                            train_split: float=1.0, 
                            classes: Optional[list]=None, 
                            target_train: Optional[int]=None, 
                            target_val: Optional[int]=None, 
                            ext: str='_spec.npy'):
    """ Prepares train/val path lists and a label dictionary

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
    """
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
    """ Resamples a list
    """
    if len(x)>target:
        x = list(np.random.choice(x, target, replace=0)) # downsample
    elif len(x)<target:
        x = x+list(np.random.choice(x, target-len(x), replace=1)) # upsample without losing information
    return x


def get_labels_pos_neg_per_class(files: list, label_map: dict) -> np.ndarray:
    """ Create a label array from the given path list
    
        Expects file path format:

            .../<'positive' or 'negative'>/<class_name>/<file_name>

    Args:
        files: list of file paths
        label_map: dictionary mapping each class in the file paths to an index
    Returns:
        y: label array; unknown values are indicated by -1
    """
    y = np.ones((len(files),len(label_map)))*-1
    for c,i in enumerate(files):
        assert i.split('/')[-3] in ('positive', 'negative'), "Error: Label could not be interpretted."
        if i.split('/')[-3]=='positive':
            y[c, int(label_map[i.split('/')[-2]])] = 1
        else:
            y[c, int(label_map[i.split('/')[-2]])] = 0
    return y 
            

def num_pos_neg_per_class(files: list):
    """ Display the number of positive and negative samples per class
    
        Expects file path format:

            .../<'positive' or 'negative'>/<class_name>/<file_name>
    Args:
        files: list of file paths
    """
    classes = []
    p = []
    n = []
    for i in list(set([i.split('/')[-2] for i in files])):
        classes.append(i)
        p.append(len(np.where([(j.split('/')[-2]==i) & (j.split('/')[-3]=='positive') for j in files])[0]))
        n.append(len(np.where([(j.split('/')[-2]==i) & (j.split('/')[-3]=='negative') for j in files])[0]))
    display(pd.DataFrame({'Class':classes,'Positives':p,'Negatives':n}))
    
    
def plot_batch_samples(batch: tf.Tensor, nr=4, nc=4):
    """ Plots a batch of examples from the DataGenerator
    
    Args:
        batch: batch tensor
        nr: number of rows to plot
        nc: number of columns to plot
    """
    plt.figure(figsize=(15,15))
    for c in range(len(batch)):
        plt.subplot(nr,nc,c+1)
        plt.pcolormesh(batch[c].numpy())
        plt.clim([-100, 20])
        plt.axis('off')
    
    
class TrainGenerator(Sequence):
    """ Generates augmented spectrogram data
    """
    def __init__(self, 
                 files: list,
                 labels: np.ndarray,
                 label_map: dict, 
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
        """
        Args:
            files:                       list of file paths
            labels:                      label array
            label_map:                   dictionary mapping classes to indices
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
        """
        'Initialization'
        self.files = tf.convert_to_tensor(files)
        self.labels = tf.convert_to_tensor(labels)
        self.label_map = label_map
        self.batch_size = batch_size
        
        self.augment = augment
        self.augconf = {}
        self.augconf['blend_prob'] = augment_blend_prob
        self.augconf['max_time_masks'] = augment_max_time_masks
        self.augconf['max_time_mask_size'] = augment_max_time_mask_size
        self.augconf['max_freq_masks'] = augment_max_freq_masks
        self.augconf['max_freq_mask_size'] = augment_max_freq_mask_size
        self.augconf['add_noise_prob'] = augment_add_noise_prob
        self.augconf['add_noise_stds'] = augment_add_noise_stds
        self.augconf['max_time_shift'] = augment_max_time_shift
        self.augconf['max_freq_shift'] = augment_max_freq_shift 
        self.augconf['max_contrast'] = augment_max_contrast
        
        self.shuffle = shuffle
        self.assume_absent = assume_absent
        assert label_format in ('multi-label', 'single-label', 'multi-class'), "Error: Unknown label format"
        self.label_format = label_format

        self.on_epoch_end()


    def __len__(self):
        """ The number of batches per epoch
        """
        return int(np.floor(len(self.files) / self.batch_size))
    
    def on_epoch_end(self):
        """ Updates indices after each epoch
        """
        self.indices = tf.range(len(self.files))
        if self.shuffle == True:
            self.indices = tf.random.shuffle(self.indices)

    def __getitem__(self, index):
        """ Wrapper for __data_generation to get batch based on index
        
        Args:
            index: batch index
        Returns:
            X: batch input tensor
            y: batch label tensor
        """
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Get batch files and labels
        files_batch = tf.gather(self.files, indices)
        labels_batch = tf.gather(self.labels, indices)
        
        # Generate data  
        X, y = self.__data_generation(files_batch, labels_batch)
        return X, y

    def __data_generation(self, files_batch, labels_batch):
        """ Generates a batch based on list of files and labels
        
        Args:
            files_batch: tensor list of file paths
            labels_batch: batch label tensor
        Returns:
            X: batch input tensor
            y: batch label tensor
        """
        y = labels_batch
        if self.assume_absent:
            y = tf.where(y==-1, tf.zeros_like(y), y)
        
        # Load raw batch
        batch = []
        for i, file in enumerate(files_batch):
            batch.append(tf.convert_to_tensor(np.load(bytes.decode(file.numpy()))))
        X = tf.stack(batch)
        X_std = tf.math.reduce_std(X)
                        
        # Augment
        if self.augment:
        
            # Blending        
            if self.augconf['blend_prob']>0:
                X, y = blend(X, 
                             y, 
                             self.label_map, 
                             prob=self.augconf['blend_prob'], 
                             label_format=self.label_format)
            
            batch = []
            for i in range(len(X)):
                
                sample = X[i,]
                
                # Contrast
                if self.augconf['max_contrast']>1:
                    sample = tf.image.random_contrast(tf.expand_dims(sample, axis=-1), 1, self.augconf['max_contrast'])[:,:,0]
                
                # Time mask
                if self.augconf['max_time_masks']>0:
                    for _ in range(tf.random.uniform([], 0, self.augconf['max_time_masks']+1, tf.int32)):
                        sample = time_mask(sample,
                                           maxwidth=tf.random.uniform([], 1, int(X.shape[2]*self.augconf['max_time_mask_size']), tf.int32))   

                # Freq mask
                if self.augconf['max_freq_masks']>0:
                    for _ in range(tf.random.uniform([], 0, self.augconf['max_freq_masks']+1, tf.int32)):
                        sample = freq_mask(sample,
                                           maxwidth=tf.random.uniform([], 1, int(X.shape[1]*self.augconf['max_freq_mask_size']), tf.int32))

                # Add noise
                if tf.random.uniform([],0,1)<self.augconf['add_noise_prob']:
                    sample = sample + tf.clip_by_value(tf.random.normal(tf.shape(sample),
                                                                        0,
                                                                        self.augconf['add_noise_stds']*X_std),
                                                       -self.augconf['add_noise_stds']*X_std,
                                                       self.augconf['add_noise_stds']*X_std)

                # Affine time-freq shift
                sample = affine_transform(sample, self.augconf['max_time_shift'], self.augconf['max_freq_shift'])
                
                batch.append(sample)
            
            X = tf.stack(batch)

        return X, y


def blend(X: tf.Tensor, y: tf.Tensor, label_map: dict, prob: float=1.0, strength: float=0.5, label_format: str='multi-label'):
    """ Apply blending augmentation to a batch of spetrograms
    
    Args:
        X: batch input tensor
        y: batch label tensor
        label_dict: dictionary mapping classes to indices
        prob: probability of blending per sample
        strength: strength of the added sample in the mix
        label_format: one of 'multi-label', 'single-label', or 'multi-class'
    Returns:
        X: blended batch input tensor
        y: blended batch label tensor
    """
    if label_format in ('multi-label', 'single-label'):
        
        toblend = tf.where(tf.random.uniform((X.shape[0], 1), 0, 1)<=prob, tf.ones_like((X.shape[0],)), tf.zeros_like((X.shape[0],)))
        toblend = tf.linalg.matmul(toblend, tf.cast(tf.ones((1, X.shape[1])), tf.int32))
        toblend = tf.repeat(tf.expand_dims(toblend, axis=-1), X.shape[2], axis=-1)
        toblend = tf.cast(toblend, tf.float32) * strength
        idxshuffle = tf.random.shuffle(tf.range(X.shape[0]))
        X = X*(1-toblend) + tf.gather(X, idxshuffle, axis=0)*toblend

        if label_format=='multi-label':
            label = combine_labels_multi(y, tf.gather(y, idxshuffle, axis=0))
        elif label_format=='single-label':
            assert y.shape[1] in (1, 2), "Error: label vector cannot be interpretted as single-label"
            if y.shape[1]==1:
                label = combine_labels_multi(y, tf.gather(y, idxshuffle, axis=0))
            elif y.shape[1]==2:
                label = combine_labels_single_one_hot(y, tf.gather(y, idxshuffle, axis=0), label_map)
    else:
        assert 'background' in label_map.keys(), \
        "Error: unsure how to blend categorical labels without a background class"
        batch = []
        labels = []
        for i,j in zip(range(X.shape[0]), tf.random.shuffle(tf.range(X.shape[0]))): # shuffle batch
            # if corresponding sample is background
            if y[j, int(label_map['background'])]==1: 
                batch.append(X[i,]*(1-strength) + X[j,]*strength) # blend it
            else:
                batch.append(X[i,])
        X = tf.stack(batch)
                     
    return X, y

def combine_labels_multi(y1: tf.Tensor, y2: tf.Tensor):
    """ Implements logic for combining multi-label vectors with compatibility for unknown labels represented by -1
    """
    y1 = tf.where(y1==1, tf.ones_like(y1)*2, y1)
    y2 = tf.where(y2==1, tf.ones_like(y2)*2, y2)
    y = y1+y2
    return tf.clip_by_value(y, -1, 1)

def combine_labels_single_one_hot(y1: tf.Tensor, y2: tf.Tensor, label_map: dict):
    """ Implements logic for combining 
        Assumes that the positive class corresponds to the 1 index
        This is how get_files_and_label_map operates
    """
    y1 = y1[:,1]
    y2 = y2[:,1]
    return tf.one_hot(tf.maximum(y1, y2))
    
def freq_mask(x: tf.Tensor, maxwidth: float):
    """ Apply frequency mask to a spectrogram
    
    From tensorflow_io
    
    Args:
        x: tensor sample
        maxwidth: maximum width (fraction of total frequency range)
    """
    limit = tf.shape(x)[0]
    t = tf.random.uniform(shape=(), minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    t0 = tf.random.uniform(
        shape=(), minval=0, maxval=limit - t, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(limit), (-1, 1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
    )
    return tf.where(condition, tf.cast(tf.reduce_mean(x), x.dtype), x)

def time_mask(x: tf.Tensor, maxwidth: float):
    """ Apply frequency mask to a spectrogram
    
    From tensorflow_io
    
    Args:
        x: tensor sample
        maxwidth: maximum width (fraction of total time)
    """
    limit = tf.shape(x)[1]
    f = tf.random.uniform(shape=(), minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    f0 = tf.random.uniform(
        shape=(), minval=0, maxval=limit - f, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(limit), (1, -1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
    )
    return tf.where(condition, tf.cast(tf.reduce_mean(x), x.dtype), x)

def affine_transform(x: tf.Tensor, time_shift_percent: float=0.0, freq_shift_percent: float=0.0):
    """ Apply affine time/frequency shifting to a spectrogram
    
    Args:
        x: tensor sample
        time_shift_percent: fraction of total time indicating max possible time shift
        freq_shift_percent: fraction of total freq range indicating max possible freq shift
    """
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
    return x[freq_shift:(freq_shift+orig_shape[0]), time_shift:(time_shift+orig_shape[1])]
    
    
    
    
    