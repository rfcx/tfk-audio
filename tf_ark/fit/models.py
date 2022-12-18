import tensorflow as tf
import tensorflow.keras.layers as layers
from . import yamnet
from ..dataprep import spec


class SpecLayer(tf.keras.layers.Layer):
    """Tensorflow Keras layer for implementing spectrogram conversion with a SpecGenerator object
    """
    def __init__(self,
                 specgen):
        super().__init__()
        self.specgen = specgen
    
    def call(self, x):
        x = self.specgen.wav_to_spec(x)
        return x 
    
class ImageNetModel(tf.keras.Model):
    """Tensorflow Keras model class for implementing pre-trained ImageNet models for audio recognition
    """
    def __init__(self, 
                 backbone, 
                 num_classes,
                 input_shape,
                 wav_input=False,
                 specgen=None,
                 hop_seconds=1.0): 
        super().__init__()
        
        # net
        self.backbone = backbone
        self.num_classes = num_classes
        self._input_shape = input_shape
        
        # preprocessing
        self.wav_input = wav_input
        self.specgen = specgen
        self.spec_layer = None
        self.scale_factor = None
        if self.specgen is not None:
            self.spec_layer = SpecLayer(self.specgen)
            # imagenet 224 compared to the raw spectrogram sample width
            self.scale_factor = 224/self.spec_layer.specgen.sample_width
        
        # prediction
        self._hop_seconds = hop_seconds
        
        self._build()

    def _build(self):        
        conv = self.backbone(weights='imagenet',
                             include_top=False,
                             input_shape=[224, 224, 3])
        for layer in conv.layers:
            layer.trainable = True
        self.conv1 = conv
        self.avgpool1 = layers.AveragePooling2D((7, 7))
        self.flatten1 = layers.Flatten()
        self.dropout1 = layers.Dropout(0.5)
        self.dense1 = layers.Dense(self.num_classes, activation='sigmoid')
        if self.wav_input:
            self.call = self.call_wav
            self.build(input_shape=(1, None,))
        else:
            self.call = self.call_spec
            self.build(input_shape=(None, *self._input_shape))
                
    def call_spec(self, x, training=False):
        x = tf.expand_dims(x, axis=-1)
        x = tf.repeat(x, 3, axis=-1)
        x = tf.image.resize(x, (224, 224))
        x = tf.image.per_image_standardization(x)
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.flatten1(x)
        x = self.dropout1(x, training=training)
        return self.dense1(x)
    
    def call_wav(self, x, training=False):
        x = self.spec_layer(x[0])
        x = tf.expand_dims(x, axis=-1)
        x = tf.repeat(x, 3, axis=-1)
        x = tf.image.resize(x, (224, tf.cast(tf.shape(x)[1], tf.float32)*self.scale_factor))
        x = tf.image.per_image_standardization(x)
        x = tf.signal.frame(x,
                            frame_length=224,
                            frame_step=int(round(self._hop_seconds * \
                                                 self.spec_layer.specgen.second_width * \
                                                 self.scale_factor)),
                            axis=1)
        x = tf.transpose(x, (1,0,2,3))
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.flatten1(x)
        x = self.dropout1(x, training=training)
        return self.dense1(x)
    
    def set_wav_input(self, wav_input, specgen=None):
        if (specgen is None) and (self.specgen is not None):
            specgen = self.specgen
        assert specgen is not None, "Error: No SpecGenerator found."            
        self.__init__(backbone=self.backbone, 
                      num_classes=self.num_classes, 
                      wav_input=wav_input, 
                      input_shape=self._input_shape,
                      specgen=specgen,
                      hop_seconds=self.hop_seconds)
        
    @property
    def hop_seconds(self):
        return self._hop_seconds
    
    @hop_seconds.setter
    def hop_seconds(self, seconds):
        self._hop_seconds = seconds
        self.make_predict_function(force=True)
        
    def summary(self):
        if self.wav_input:
            x = layers.Input(batch_shape=(1, None,))
        else:
            x = layers.Input(batch_shape=(None, *self._input_shape))
        model = tf.keras.Model(inputs=x, outputs=self.call(x))
        return model.summary()


    
    
    
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
    
        
