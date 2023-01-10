import tensorflow as tf
import tensorflow.keras.layers as layers
from . import yamnet
from ..dataprep import spec
import tensorflow.keras.applications as imagenet_models


def imagenet_audio_model(backbone, num_classes, specgenerator, hop_seconds=1.0):
    ''' Creates a Tensorflow model for predicting on audio waveforms
    
    Args:
        backbone:        imagenet model backbone, from tf.keras.applications
        num_classes:     number of classes
        specgenerator:   tfk.dataprep.spec.SpecGenerator object
        hop_seconds:     number of seconds between predictions
    Returns
        A Tensorflow model
    '''    
    conv = backbone(weights='imagenet', include_top=False, input_shape=[224, 224, 3])
    for layer in conv.layers:
        layer.trainable = True
        
    specparams = {k:v for (k,v) in specgenerator.__dict__.items() if k!='_processed_files'}
    scale_factor = 224/specparams['sample_width']

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
            name='spec1'
        )(inputs)
    x = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            name='expand1'
        )(x)
    x = layers.Lambda(
            lambda x: tf.image.resize(x, (tf.cast(tf.shape(x)[0], tf.float32)*scale_factor, 224)),
            name='resize1'
        )(x)
    x = layers.Lambda(
            lambda x: tf.keras.backend.repeat_elements(x=x, rep=3, axis=-1),
            name='repeat1'
        )(x)
    x = layers.Lambda(
            lambda x: tf.image.per_image_standardization(x),
            name='norm1'
        )(x)
    x = layers.Lambda(
            lambda x: tf.signal.frame(x,
                                     frame_length=224,
                                     frame_step=int(round(hop_seconds * \
                                                          specparams['second_width'] * \
                                                          scale_factor)),
                                     axis=0),
            name='frame1'
        )(x)
    x = conv(x)
    x = layers.AveragePooling2D((7, 7))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    
def imagenet_spec_model(backbone, num_classes, input_shape):
    ''' Creates a Tensorflow model for predicting on spectrograms
    
    Args:
        backbone:        imagenet model backbone, from tf.keras.applications
        num_classes:     number of classes
        input_shape:     input spectrogram (height, width)
    Returns
        A Tensorflow model
    '''
    conv = backbone(weights='imagenet', include_top=False, input_shape=[224, 224, 3])
    for layer in conv.layers:
        layer.trainable = True
    
    inputs = layers.Input(input_shape)
    x = layers.Reshape(target_shape=(*input_shape, 1))(inputs)
    x = layers.Lambda(
            lambda x: tf.image.resize(x, (224, 224)),
            name='resize1'
        )(x)
    x = layers.Lambda(
            lambda x: tf.keras.backend.repeat_elements(x=x, rep=3, axis=3)
        )(x)
    x = layers.Lambda(
            lambda x: tf.image.per_image_standardization(x)
        )(x)
    x = conv(x)
    x = layers.AveragePooling2D((7, 7))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)

                               
    
    
    
# class YAMNet(tf.keras.Model):
#     def __init__(self, config):    
#         super().__init__()
        
#         self._config = config
        
#         class params:
#             patch_bands = config['spec']['mel_bands']
#             patch_frames = int(config['audio']['sample_seconds']/config['spec']['stft_hop_seconds']-(config['spec']['stft_window_seconds']/config['spec']['stft_hop_seconds'])+1)
#             num_classes = config['fit']['output_nodes']
#             classifier_activation = config['fit']['output_activation']
#             # rest of parameters left to YAMNet defaults
#             conv_padding = 'same'
#             batchnorm_center = True
#             batchnorm_scale = False
#             batchnorm_epsilon = 1e-4
            
#         self.conv1 = yamnet_conv_model(params)
#         self.dense1 = layers.Dense(64, activation='relu')
#         self.dropout1 = layers.Dropout(config['fit']['dropout'])
#         self.dense2 = layers.Dense(config['fit']['output_nodes'], activation=config['fit']['output_activation'])
#         self.build(input_shape=self.conv1.input_shape)
        
#     def call(self, inputs, training=False):
#         x = self.conv1(inputs)
#         x = self.dense1(x)
#         x = self.dropout1(x)
#         return self.dense2(x)
    
#     def summary(self):
#         x = layers.Input(self.conv1.input_shape[1:])
#         model = tf.keras.Model(inputs=x, outputs=self.call(x))
#         return model.summary()
    

# def yamnet_conv_model(params):
#     specs = layers.Input(batch_shape=(None, params.patch_frames, params.patch_bands), dtype=tf.float32)
#     _, embeddings = yamnet.yamnet(specs, params)
#     model = tf.keras.Model(inputs=specs, outputs=embeddings)
#     return model
    
        
