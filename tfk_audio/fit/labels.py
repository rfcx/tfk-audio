import os
import numpy as np
import pandas as pd
from typing import Optional, Callable


def get_classification_labels(files: list, label_map: dict, label_format: str) -> np.ndarray:
    ''' Create label vectors for multi-label, multi-class, or binary classification
        
    You can choose from label formats:

        multi-label
        multi-class-one-hot
        multi-class-categorical
        binary


        ---- multi-label ----

            example:     y = [[0, 1, 0, -1],     [<# samples>, <# classes>]
                              [1, -1, 0, 0]]
                              
            In this case, the code will search for files with the same path but with an added .csv extension,
                these files should contain a single column containing the list of present class names, with no header.
            Otherwise, if the files are in a directory with structure 

                    .../<"positive" or "negative">/<class_name>/<filename>

                then each label vector will contain a 1 or 0 at the index of the labeled species,
                and -1's elsewhere, to indicate unknown labels.
                
            Unknown labels can be indicated by -1 and used with a loss function wrapped with tfk_audio.fit.loss.mask_loss().

        ---- multi-class-one-hot ----

            example:     y = [[0, 0, 0, 1],     [<# samples>, <# classes>]
                              [0, 1, 0, 0]]

            File paths are assumed to follow .../<class_name>/<filename>

        ---- multi-class-categorical ----

            example:     y = [[3],              [<# samples>, 1] 
                              [1]]

            File paths are assumed to follow .../<class_name>/<filename>

        ---- binary ----

            example:     y = [[0],              [<# samples>, 1] 
                              [0], 
                              [1], 
                              [1]]
            
    
    Args:
        files:        list of data file paths
        label_map:    dictionary mapping class names to label indices
        label_format: target label format
        
    Returns:
        y:            label vector
    '''
    
    # check label format
    assert label_format in ('multi-label', 'multi-class-one-hot', 'multi-class-categorical', 'binary'), 'Error: Unknown label format'
    
    # multi-label
    if label_format=='multi-label':
        assert (os.path.exists(files[0]+'.csv')) or (files[0].split('/')[-3] in ('positive', 'negative')), \
        'Error: Cannot determine how to read labels in multi-label format.'
        if os.path.exists(files[0]+'.csv'):
            y = get_labels_multi_label_csv(files, label_map)
        else:
            y = get_labels_pn_per_class(files, label_map)
    elif label_format=='multi-class-one-hot':
        y = get_labels_multi_class_one_hot(files, label_map)
    elif label_format=='multi-class-categorical':
        y = get_labels_multi_class_categorical(files, label_map)
    
    else:
        y= get_labels_binary(files, label_map)
        
    return y


def get_labels_binary(files: list, label_map: dict) -> np.ndarray:
    assert len(label_map)>0, 'Error: Empty label map'
    assert len(label_map)<3, 'Error: Too many classes to create binary labels'
    if len(label_map)>1:
        assert 'background' in list(label_map.keys()), 'Error: Could not interpret label_map as binary. \
        It should contain a single class, or two classes with one named "background".'  
    if len(label_map)==1:
        assert files[0].split('/')[-3] in ('positive', 'negative'), \
        'Error: Unsure how to determine if the class is present. If label_map has a single class, \
        files should be in a directory that follows <"positive" or "negative">/<class_name>/<filename>. \
        Otherwise, it should have two classes with one named "background".'
    y = np.ones((len(files),1))*-1
    for c,i in enumerate(files):
        if len(label_map)==1:
            label = i.split('/')[-3]
            assert label in ('positive', 'negative'), \
            'Error: Could not interpret label. If label_map has a single class, \
            files should be in a directory that follows <"positive" or "negative">/<class_name>/<filename>.'
            if label=='positive':
                y[c] = 1
            else:
                y[c] = 0
        else:
            clas = i.split('/')[-2]
            y[c] = label_map[clas]
    return y


def get_labels_multi_class_one_hot(files: list, label_map: dict) -> np.ndarray:
    y = np.zeros((len(files),len(label_map)))
    for c,i in enumerate(files):
        assert i.split('/')[-2] in list(label_map.keys()), 'Error: Class not found in label map.'
        y[c, int(label_map[i.split('/')[-2]])] = 1
    return y


def get_labels_multi_class_categorical(files: list, label_map: dict) -> np.ndarray:
    y = np.zeros((len(files),1))
    for c,i in enumerate(files):
        assert i.split('/')[-2] in list(label_map.keys()), 'Error: Class not found in label map.'
        y[c] = int(label_map[i.split('/')[-2]])
    return y


def get_labels_pn_per_class(files: list, label_map: dict) -> np.ndarray:
    y = np.ones((len(files),len(label_map)))*-1
    for c,i in enumerate(files):
        assert i.split('/')[-3] in ('positive', 'negative'), 'Error: Label could not be interpretted.'
        if i.split('/')[-3]=='positive':
            y[c, int(label_map[i.split('/')[-2]])] = 1
        else:
            y[c, int(label_map[i.split('/')[-2]])] = 0
    return y


def get_labels_multi_label_csv(files: list, label_map: dict) -> np.ndarray:
    y = np.zeros((len(files),len(label_map)))
    for c,i in enumerate(files):
        assert os.path.exists(i+'.csv'), 'Error: Could not find label CSV for '+i
        if os.stat(i+'.csv').st_size == 0:
            continue
        else:
            classes = list(pd.read_csv(i+'.csv', header=None)[0])
            for j in classes:
                y[c, int(label_map[j])] = 1
    return y
