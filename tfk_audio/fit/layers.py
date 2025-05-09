import inspect
import tensorflow as tf
import tensorflow.keras.layers as layers
from ..preprocess import audio, spec
    


def SpecImageNet(target_shape,
                 image_scaling=True,
                 channels=3):
    ''' Returns a model to convert spectrogram inputs to ImageNet inputs
    '''
    args = locals()
    inputs = layers.Input((None, None))
    outputs = layers.Lambda(spec_to_imagenet,
                            arguments=args)(inputs)
    return tf.keras.models.Model(inputs, outputs, name='spec_to_imagenet')
                   
    
def WavImageNet(target_shape,
                spec_params,
                image_scaling=True,
                channels=3):
    ''' Returns a model to convert waveform inputs to ImageNet inputs
    '''
    args = locals()
    inputs = layers.Input(int(spec_params['sample_rate']*spec_params['sample_seconds']))
    outputs = layers.Lambda(wav_to_imagenet,
                            arguments=args)(inputs)
    return tf.keras.models.Model(inputs, outputs, name='wav_to_imagenet')


def spec_to_imagenet(inputs, target_shape, image_scaling=True, scaling_factor=255, channels=3):
    '''
    Converts a batch of 2D spectrogram inputs to 3-channel images.

    Args:
        inputs: Tensor of shape (batch_size, height, width) or (batch_size, height, width, channels).
        target_shape: Tuple (height, width) specifying the target image size.
        image_scaling: Boolean indicating whether to scale the image to [0, 255].
        channels: Number of output channels (e.g., 1 for grayscale, 3 for RGB).

    Returns:
        Tensor of shape (batch_size, target_height, target_width, channels).
    '''
    if (len(inputs.shape)==3): # if no channels axis already
        x = inputs[..., tf.newaxis]
    else:
        x = inputs
            
    # resize image
    x = layers.Resizing(target_shape[0], target_shape[1])(x)

    # add channels
    if channels > 1:
        x = tf.repeat(x, repeats=channels, axis=-1)
    elif channels == 0:
        x = x[..., 0]
        
    # scale
    if image_scaling:
        x = tf.map_fn(norm, x)
        x *= scaling_factor
        
    return x


def wav_to_imagenet(inputs, target_shape, spec_params, image_scaling=True, channels=3):
    ''' Converts a 1D waveform to a 3-channel image
    
    Args:
        inputs: Tensor of shape (batch_size, time).
        target_shape: Tuple (height, width) specifying the target image size.
        spec_params: Dictionary of parameters for spectrogram conversion.
        image_scaling: Boolean indicating whether to scale the image to [0, 255].
        channels: Number of output channels (e.g., 1 for grayscale, 3 for RGB).
    Returns:
        Tensor of shape (batch_size, target_height, target_width, channels).
    '''
    x = layers.Lambda(
        lambda i: tf.map_fn(lambda j:
                            spec._wav_to_spec(j,
                                              sample_rate = spec_params['sample_rate'],
                                              stft_window_samples = spec_params['stft_window_samples'],
                                              stft_hop_samples = spec_params['stft_hop_samples'],
                                              min_hz = spec_params['min_hz'],
                                              max_hz = spec_params['max_hz'],
                                              fft_length = spec_params['fft_length'],
                                              db_limits = spec_params['db_limits'],
                                              db_scale = spec_params['db_scale'],
                                              mel_bands = spec_params['mel_bands'],
                                              normalize_audio = spec_params['norm'],
                                              normalize_rms_db = spec_params['norm_db']
                                            ),
                            i),
        name='spec'
        )(inputs)
    return spec_to_imagenet(x, 
                            target_shape=target_shape, 
                            image_scaling=image_scaling, 
                            channels=channels)


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


# draft - enabling audio windowing within the WavImageNet layer
# def wav_to_imagenet(inputs,
#                     target_shape,
#                     spec_params,
#                     hop_seconds = 1.0,
#                     image_scaling = True):
#     ''' Converts a 1D waveform to a 3-channel image
    
#     '''   
#     scale_factor = target_shape[0]/spec_params['sample_width']
    
#     x = layers.Lambda(
#             lambda x: spec._wav_to_spec(x[0,],
#                                         sample_rate = spec_params['sample_rate'],
#                                         stft_window_samples = spec_params['stft_window_samples'],
#                                         stft_hop_samples = spec_params['stft_hop_samples'],
#                                         min_hz = spec_params['min_hz'],
#                                         max_hz = spec_params['max_hz'],
#                                         fft_length = spec_params['fft_length'],
#                                         db_scale = spec_params['db_scale'],
#                                         db_limits = spec_params['db_limits'],
#                                         mel_bands = spec_params['mel_bands'],
#                                         tflite_compatible = spec_params['tflite_compatible']
#                                         ),
#             name='spec'
#         )(inputs)
#     x = x[..., tf.newaxis]
#     print(x.shape)
    
#     # resize image
#     x = layers.Resizing(tf.cast(tf.shape(x)[0], tf.float32)*scale_factor, target_shape[1])(x)
#     print(x.shape)
    
#     x = tf.tile(x, [1, 1, 3])
    
#     # divide spectrogram into windows
#     x = layers.Lambda(
#             lambda x: tf.signal.frame(x,
#                                       frame_length=target_shape[0],
#                                       frame_step=int(round(hop_seconds * \
#                                                            spec_params['second_width'] * \
#                                                            scale_factor)),
#                                       axis=0),
#             name='frame'
#         )(x)
    
#     # scale
#     if image_scaling:
#         x = tf.map_fn(norm, x)
#         x *= 255
        
#     return x










    
        
