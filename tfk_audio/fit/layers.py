import inspect
import tensorflow as tf
import tensorflow.keras.layers as layers
from . import yamnet
from .. import dataprep
    


def SpecToImageNet(target_shape,
                   image_scaling=True):
    args = locals()
    inputs = layers.Input((None, None))
    outputs = layers.Lambda(spec_to_imagenet,
                            arguments=args)(inputs)
    return tf.keras.models.Model(inputs, outputs, name='spec_to_imagenet')
                   
    
def WavToImageNet(target_shape,
                  spec_params,
                  image_scaling=True):
    args = locals()
    inputs = layers.Input((None,))
    outputs = layers.Lambda(wav_to_imagenet,
                            arguments=args)(inputs)
    return tf.keras.models.Model(inputs, outputs, name='wav_to_imagenet')


def spec_to_imagenet(inputs, target_shape, image_scaling=True):
    ''' Converts a batch of 2D inputs to 3-channel images
    
    '''
    if len(inputs.shape)==3: # if no channels axis already
        x = inputs[..., tf.newaxis]
    else:
        x = inputs
            
    # resize image
    x = layers.Resizing(target_shape[0], target_shape[1])(x)

    # add channels
    x = tf.tile(x, [1, 1, 1, 3])

    # scale
    if image_scaling:
        x = norm(x)
        x *= 255
        
    return x


def wav_to_imagenet(inputs,
                    target_shape,
                    spec_params,
                    hop_seconds = 1.0,
                    image_scaling = True):
    ''' Converts a 1D waveform to a 3-channel image
    
    '''   
    scale_factor = target_shape[0]/spec_params['sample_width']
    
    x = layers.Lambda(
            lambda x: dataprep.spec._wav_to_spec(x[0,],
                                                 sample_rate = spec_params['sample_rate'],
                                                 stft_window_samples = spec_params['stft_window_samples'],
                                                 stft_hop_samples = spec_params['stft_hop_samples'],
                                                 fft_length = spec_params['fft_length'],
                                                 mel_bands = spec_params['mel_bands'],
                                                 min_hz = spec_params['min_hz'],
                                                 max_hz = spec_params['max_hz'],
                                                 db_scale = spec_params['db_scale'],
                                                 db_limits = spec_params['db_limits'],
                                                 tflite_compatible = spec_params['tflite_compatible']
                                                 ),
            name='spec'
        )(inputs)
    x = x[..., tf.newaxis]
    
    # resize image
    x = layers.Resizing(tf.cast(tf.shape(x)[0], tf.float32)*scale_factor, target_shape[1])(x)
    
    x = tf.tile(x, [1, 1, 3])
    
    # divide spectrogram into windows
    x = layers.Lambda(
            lambda x: tf.signal.frame(x,
                                      frame_length=target_shape[0],
                                      frame_step=int(round(hop_seconds * \
                                                           spec_params['second_width'] * \
                                                           scale_factor)),
                                      axis=0),
            name='frame'
        )(x)
    
    # scale
    if image_scaling:
        x = norm(x)
        x *= 255
        
    return x


def norm(tensor):
    ''' Normalizes input to [-1, 1]
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
    return tensor









    
        
