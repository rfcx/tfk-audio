import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

def get_files_and_label_dict(data_dirs, train_split=0.8, num_per_class=None, classes=None, ext='_spec.npy'):
    """
    Get the list of training files and a label dictionary for the classes
    Optionally split files into train and validation sets stratified by class
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
    labels = dict()
    if ('background' in classes):
        classes = ['background']+[i for i in classes if i!='background']
    for cnt, i in enumerate(classes): # loop over classes
        labels[i]=cnt # add class to dict
        
        # get class files
        class_files=[]
        for dr in data_dirs:
            if not os.path.exists(dr+i):
                continue
            class_files += [dr+i+'/'+j for j in os.listdir(dr+i) if j.endswith(ext)]
        
        # maybe resample
        if num_per_class is not None:
            if len(class_files)>num_per_class:
                class_files = list(np.random.choice(class_files, num_per_class, replace=0)) # downsample
            elif len(class_files)<num_per_class:
                class_files = class_files+list(np.random.choice(class_files,
                                                                num_per_class-len(class_files), 
                                                                replace=1)) # upsample
        # split data
        if train_split==1.0:
            class_train=files
            class_val=[]
        else:
            class_train, class_val = train_test_split(class_files,
                                                      train_size=train_split)
            
        files_train += class_train
        files_val += class_val
                
    return files_train, files_val, labels

def get_labels_pos_neg_per_class(files, labels):
    """ Create a list of labels associated with each file
    
            Expects file path format:
                ".../<"positive" or "negative">/<"class_name">/<"file_name">                
    """
    y = np.ones((len(files),len(labels)))*-1
    for c,i in enumerate(files):
        assert i.split('/')[-3] in ('positive', 'negative'), "Error: Label could not be interpretted."
        if i.split('/')[-3]=='positive':
            y[c, int(labels[i.split('/')[-2]])] = 1
        else:
            y[c, int(labels[i.split('/')[-2]])] = 0
    return y 
            

def num_pos_neg_per_class(files):
    """ Display the number of positive and negative samples per class
    
            Expects file path format:
                ".../<"positive" or "negative">/<"class_name">/<"file_name">               
    """
    classes = []
    p = []
    n = []
    for i in list(set([i.split('/')[-2] for i in files])):
        classes.append(i)
        p.append(len(np.where([(j.split('/')[-2]==i) & (j.split('/')[-3]=='positive') for j in files])[0]))
        n.append(len(np.where([(j.split('/')[-2]==i) & (j.split('/')[-3]=='negative') for j in files])[0]))
    display(pd.DataFrame({'Class':classes,'Positives':p,'Negatives':n}))
    
    
def plot_batch_samples(batch, nr=4, nc=4):
    """ Plots a batch of examples from the DataGenerator
    """
    plt.figure(figsize=(10,10))
    for c in range(len(batch)):
        plt.subplot(nr,nc,c+1)
        plt.pcolormesh(batch[c].numpy());
        plt.axis('off')
    
    
class TrainGenerator(Sequence):
    """ Generates augmented spectrogram data
    """
    def __init__(self, 
                 files,
                 labels,
                 label_dict, 
                 batch_size = 1,
                 augment = True,
                 augment_blend_prob = 0,
                 augment_max_time_masks = 5,
                 augment_max_time_mask_size = 0.33,
                 augment_max_freq_masks = 5,
                 augment_max_freq_mask_size = 0.05,
                 augment_add_noise_prob = 0.33,
                 augment_add_noise_stds = 1.0,
                 augment_max_time_shift = 0.33,
                 augment_max_freq_shift = 0.05,
                 shuffle = True,
                 assume_absent = True,
                 label_format = 'multi-label'):
        'Initialization'
        self.files = tf.convert_to_tensor(files)
        self.labels = tf.convert_to_tensor(labels)
        self.label_dict = label_dict
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
        
        self.shuffle = shuffle
        self.assume_absent = assume_absent
        assert label_format in ('multi-label', 'single-label', 'multi-class'), "Error: Unknown label format"
        self.label_format = label_format

        self.on_epoch_end()


    def __len__(self):
        """ Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        """ Wrapper for __data_generation to get batch based on index
        """
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Get batch files and labels
        files_batch = tf.gather(self.files, indices)
        labels_batch = tf.gather(self.labels, indices)
        
        # Generate data  
        X, y = self.__data_generation(files_batch, labels_batch)
        return X, y

    def on_epoch_end(self):
        """ Updates indices after each epoch
        """
        self.indices = tf.range(len(self.files))
        if self.shuffle == True:
            self.indices = tf.random.shuffle(self.indices)

    def __data_generation(self, files_batch, labels_batch):
        """ Generates a batch based on list of files and labels
        """
        # Setup
        y = labels_batch
        if self.assume_absent:
            y = tf.where(y==-1, tf.zeros_like(y), y)
        
        # Load batch
        batch = []
        for i, file in enumerate(files_batch):
            batch.append(tf.convert_to_tensor(np.load(bytes.decode(file.numpy()))))
        X = tf.stack(batch)
        X_mean = tf.reduce_mean(X)
        X_std = tf.math.reduce_std(X)
                        
        # Augment
        if self.augment:
        
            # Blending        
            if self.augconf['blend_prob']>0:
                X, y = blend(X, 
                             y, 
                             self.label_dict, 
                             prob=self.augconf['blend_prob'], 
                             label_format=self.label_format)
            
            batch = []
            for i in range(len(X)):
                
                sample = X[i,]
                
                # Time mask
                if self.augconf['max_time_masks']>0:
                    for _ in range(tf.random.uniform([], 0, self.augconf['max_time_masks'], tf.int32)):
                        sample = time_mask(sample,
                                           param=tf.random.uniform([], 1, int(X.shape[2]*self.augconf['max_time_mask_size']), tf.int32))   

                # Freq mask
                if self.augconf['max_freq_masks']>0:
                    for _ in range(tf.random.uniform([], 0, self.augconf['max_freq_masks'], tf.int32)):
                        sample = freq_mask(sample,
                                           param=tf.random.uniform([], 1, int(X.shape[1]*self.augconf['max_freq_mask_size']), tf.int32))

                # Add noise
                if tf.random.uniform([],0,1)<self.augconf['add_noise_prob']:
                    sample = sample + tf.clip_by_value(tf.random.normal(tf.shape(sample),
                                                                        X_mean,
                                                                        self.augconf['add_noise_stds']*X_std),
                                                       X_mean-self.augconf['add_noise_stds']*X_std,
                                                       X_mean+self.augconf['add_noise_stds']*X_std)

                # Affine time-freq shift
                sample = affine_transform(sample, self.augconf['max_time_shift'], self.augconf['max_freq_shift'])
                
                batch.append(sample)
            
            X = tf.stack(batch)

        return X, y


def blend(X, y, label_dict, prob=1.0, strength=0.5, label_format='multi-label'):
    """
    Apply blending augmentation to a batch of spetrograms
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
                label = combine_labels_single_one_hot(y, tf.gather(y, idxshuffle, axis=0), label_dict)
    else:
        assert 'background' in label_dict.keys(), \
        "Error: unsure how to blend categorical labels without a background class"
        batch = []
        labels = []
        for i,j in zip(range(X.shape[0]), tf.random.shuffle(tf.range(X.shape[0]))): # shuffle batch
            # if corresponding sample is background
            if y[j, int(label_dict['background'])]==1: 
                batch.append(X[i,]*(1-strength) + X[j,]*strength) # blend it
            else:
                batch.append(X[i,])
        X = tf.stack(batch)
                     
    return X, y

def combine_labels_multi(y1, y2):
    """ Implements logic for combining multi-label vectors with compatibility for unknown labels represented by -1
    """
    y1 = tf.where(y1==1, tf.ones_like(y1)*2, y1)
    y2 = tf.where(y2==1, tf.ones_like(y2)*2, y2)
    y = y1+y2
    return tf.clip_by_value(y, -1, 1)

def combine_labels_single_one_hot(y1, y2, label_dict):
    """ Implements logic for combining 
        Assumes that the positive class corresponds to the 1 index
        This is how get_files_and_label_dict operates
    """
    y1 = y1[:,1]
    y2 = y2[:,1]
    return tf.one_hot(tf.maximum(y1, y2))
    
def freq_mask(inpt, param):
    """ Apply frequency masks to a spectrogram
    From tensorflow_io
    """
    time_max = tf.shape(inpt)[0]
    t = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
    t0 = tf.random.uniform(
        shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(time_max), (-1, 1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
    )
    return tf.where(condition, tf.cast(tf.reduce_mean(inpt), inpt.dtype), inpt)

def time_mask(inpt, param):
    """ Apply time masks to a spectrogram
    From tensorflow_io
    """
    freq_max = tf.shape(inpt)[1]
    f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
    f0 = tf.random.uniform(
        shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(freq_max), (1, -1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
    )
    return tf.where(condition, tf.cast(tf.reduce_mean(inpt), inpt.dtype), inpt)

def affine_transform(inpt, time_shift_percent=0, freq_shift_percent=0):
    """ Apply affine time/frequency shifting to a spectrogram
    """
    orig_shape = inpt.shape
    if time_shift_percent>0:
        time_pad = tf.random.uniform([], 0, int(inpt.shape[1]*time_shift_percent), tf.int32)
    else:
        time_pad = 0
    if freq_shift_percent>0:
        freq_pad = tf.random.uniform([], 0, int(inpt.shape[0]*freq_shift_percent), tf.int32)
    else:
        freq_pad = 0
    inpt = tf.pad(inpt, [[freq_pad, freq_pad], [time_pad, time_pad]], mode='CONSTANT', constant_values=tf.reduce_mean(inpt))
    if time_pad>0:
        time_shift = tf.random.uniform([], 0, int(time_pad*2), tf.int32)
    else:
        time_shift = 0
    if freq_pad>0:
        freq_shift = tf.random.uniform([], 0, int(freq_pad*2), tf.int32)
    else:
        freq_shift = 0
    return inpt[freq_shift:(freq_shift+orig_shape[0]), time_shift:(time_shift+orig_shape[1])]
    
    
    
    
    