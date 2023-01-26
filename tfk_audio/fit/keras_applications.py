import inspect
import tensorflow as tf
import tensorflow.keras.layers as layers
from . import yamnet
from ..dataprep import spec


    
def tf_norm(tensor):
    ''' Normalizes a tensor using the 'tf' convention in tf.keras.applications
        Normalizes input to [-1, 1]
    '''
    tensor = tf.divide(
       tf.subtract(
          tensor, 
          tf.reduce_min(tensor)
       ), 
       tf.subtract(
          tf.reduce_max(tensor), 
          tf.reduce_min(tensor)
       )
    )
    tensor *= 2
    tensor -= 1
    return tensor

def default_norm(tensor):
    ''' Normalizes input to [0, 255]
    '''
    tensor = tf.divide(
       tf.subtract(
          tensor, 
          tf.reduce_min(tensor)
       ), 
       tf.subtract(
          tf.reduce_max(tensor), 
          tf.reduce_min(tensor)
       )
    )
    tensor *= 255
    return tensor

def torch_norm(tensor):
    ''' Normalizes a tensor using the 'torch' convention in tf.keras.applications
        Normalizes input to [0, 1], and then scales based on ImageNet dataset
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = tf.divide(
       tf.subtract(
          tensor, 
          tf.reduce_min(tensor)
       ), 
       tf.subtract(
          tf.reduce_max(tensor), 
          tf.reduce_min(tensor)
       )
    )
    tensor = tf.subtract(tensor, mean)
    tensor = tf.divide(tensor, std)
    return tensor

def caffe_norm(tensor):
    ''' Normalizes a tensor using the 'caffe' convention in tf.keras.applications
        Centers based on ImageNet dataset
    '''
    mean = [103.939, 116.779, 123.68]
    tensor = default_norm(tensor)
    tensor = tf.subtract(tensor, mean)
    return tensor


def get_model_preprocess_mode(base_model):
    ''' Get the preprocessing mode of a Keras Applications model function
    
    Args:
        base_model:     Keras Applications model function
    Returns
        One of 'tf', 'torch', 'caffe', or None        
            
    '''
    # if there is no keras.applications submodule for the model
    # there is no preprocess_input function
    if max([len(i.split('.')) for i in base_model._keras_api_names])<4:
        return None
    
    # get the actual preprocessing function
    preprocess_fn = getattr(
        getattr(
            tf.keras.applications, base_model._keras_api_names[0].split('.')[-2]
        ), 
        'preprocess_input'
    )
    
    # get the source code
    preprocess_src = inspect.getsource(preprocess_fn)
    # if it uses the generic preprocess_input function, parse the mode
    if 'imagenet_utils.preprocess_input' in preprocess_src:
        preprocess_mode = preprocess_src.split('mode=')[-1].split(',')[0].strip(" )")
    else:
        preprocess_mode = None
        
    return preprocess_mode


def spec_to_rgb_image(input_shape,
                      target_shape,
                      image_scaling=True):
    
    inputs = layers.Input(input_shape, name='input')
    
    # add channels axis
    x = layers.Reshape(target_shape=(*input_shape, 1))(inputs)
    
    # repeat channels
    x = layers.Lambda(
            lambda i: tf.image.resize(i, target_shape),
            name='resize'
        )(x)
    
    x = layers.Lambda(
            lambda i: tf.keras.backend.repeat_elements(x=i, rep=3, axis=3),
            name='repeat'
        )(x)
    
    if image_scaling:
        x *= 255
        
    
        
    


def keras_applications_spec_model(base_model, 
                                  num_classes, 
                                  input_shape, 
                                  target_shape,
                                  base_model_preprocessor = None,
                                  activation = 'sigmoid'
                                 ):
    ''' Adapts a Keras Applications model for spectrogram recognition
    
    Args:
        base_model:               a function to load a pre-trained imagenet model
        num_classes:              number of classes
        target_shape:             target input image shape
        base_model_preprocessor:  preprocess_input function for the model; must be given for non keras apps models
    
    Returns
        A Tensorflow model
    '''
    conv = base_model(weights='imagenet', 
                      include_top=False, 
                      input_shape=[*target_shape, 3])
    conv._name = 'base_model'
    
    preprocess_mode = get_model_preprocess_mode(base_model)
    
#     preprocess_fn = getattr(
#         getattr(
#             tf.keras.applications, base_model._keras_api_names[0].split('.')[-2]
#         ), 
#         'preprocess_input'
#     )
    
    inputs = layers.Input(input_shape, name='input')
    
    x = layers.Reshape(target_shape=(*input_shape, 1))(inputs)
    
    x = layers.Lambda(
            lambda i: tf.image.resize(i, target_shape),
            name='resize'
        )(x)
    
    x = layers.Lambda(
            lambda i: tf.keras.backend.repeat_elements(x=i, rep=3, axis=3),
            name='repeat'
        )(x)
        
    if preprocess_mode=='tf':
        x = layers.Lambda(lambda i: tf_norm(i), name='preprocess')(x)
        
    elif preprocess_mode=='torch':
        x = layers.Lambda(lambda i: torch_norm(i), name='preprocess')(x)
        
    elif preprocess_mode=='caffe':
        x = layers.Lambda(lambda i: caffe_norm(i), name='preprocess')(x)
        
    elif preprocess_mode is None:
        x = layers.Lambda(lambda i: default_norm(i), name='preprocess')(x)

#     x = layers.Lambda(
#             lambda i: preprocess_fn(i),
#             name='preprocess'
#         )(x)
    
    x = conv(x, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation=activation)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
    return model
    


def keras_applications_audio_model(base_model, 
                                   num_classes, 
                                   specgenerator, 
                                   target_shape,
                                   hop_seconds = 1.0,
                                   base_model_preprocessor = None,
                                   activation = 'sigmoid'):
    ''' Adapts a Keras Applications model for audio recognition
    
    Args:
        base_model:        imagenet model base_model, from tf.keras.applications
        num_classes:     number of classes
        specgenerator:   tfk.dataprep.spec.SpecGenerator object
        hop_seconds:     number of seconds between predictions
    Returns
        A Tensorflow model
    '''    
    conv = base_model(weights='imagenet', 
                      include_top=False, 
                      input_shape=[*target_shape, 3])
    conv._name = 'base_model'
    
    preprocess_fn = getattr(
        getattr(
            tf.keras.applications, base_model._keras_api_names[0].split('.')[-2]
        ), 
        'preprocess_input'
    )
    preprocess_src = inspect.getsource(preprocess_fn)
    if 'imagenet_utils.preprocess_input' in preprocess_src:
        preprocess_mode = preprocess_src.split('=')[-1].strip("')\n")
    else:
        preprocess_mode = None
        
    specparams = {k:v for (k,v) in specgenerator.__dict__.items() if k!='_processed_files'}
    scale_factor = target_shape[0]/specparams['sample_width']

    inputs = layers.Input(batch_shape=(1,None,))   
    
    x = layers.Lambda(
            lambda x: spec._wav_to_spec(x[0,],
                                        sample_rate = specparams['sample_rate'],
                                        stft_window_samples = specparams['stft_window_samples'],
                                        stft_hop_samples = specparams['stft_hop_samples'],
                                        fft_length = specparams['fft_length'],
                                        mel_bands = specparams['mel_bands'],
                                        min_hz = specparams['min_hz'],
                                        max_hz = specparams['max_hz'],
                                        db_scale = specparams['db_scale'],
                                        db_limits = specparams['db_limits'],
                                        tflite_compatible = specparams['tflite_compatible']
                                        ),
            name='spec'
        )(inputs)
    
    x = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            name='reshape'
        )(x)
    
    x = layers.Lambda(
            lambda x: tf.image.resize(x, (tf.cast(tf.shape(x)[0], tf.float32)*scale_factor, target_shape[1])),
            name='resize'
        )(x)
    
    x = layers.Lambda(
            lambda x: tf.keras.backend.repeat_elements(x=x, rep=3, axis=-1),
            name='repeat'
        )(x)
    
    if preprocess_mode=='tf':
        x = layers.Lambda(lambda i: tf_norm(i), name='preprocess')(x)
        
    elif preprocess_model=='torch':
        x = layers.Lambda(lambda i: torch_norm(i), name='preprocess')(x)
        
    elif preprocess_model=='caffe':
        x = layers.Lambda(lambda i: caffe_norm(i), name='preprocess')(x)
        
    elif preprocess_model=='none':
        x = layers.Lambda(lambda i: default_norm(i), name='preprocess')(x)
    
    x = layers.Lambda(
            lambda x: tf.signal.frame(x,
                                      frame_length=target_shape[0],
                                      frame_step=int(round(hop_seconds * \
                                                           specparams['second_width'] * \
                                                           scale_factor)),
                                      axis=0),
            name='frame1'
        )(x)
    
    x = conv(x, training=False)
    
    x = layers.AveragePooling2D((7, 7))(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation=activation)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
    
        
