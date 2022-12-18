import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.sys.path.append('../')
from tfk_audio.fit import datagen, models
import tensorflow as tf
import tensorflow.keras.applications as imagenet_models

# get a list of train/val files and create label dictionary
files_train, files_val, labels = datagen.get_files_and_label_map(['./tmp/positive/', './tmp/negative/'])

# check number of samples per class
datagen.num_pos_neg_per_class(files_train)
datagen.num_pos_neg_per_class(files_val)

