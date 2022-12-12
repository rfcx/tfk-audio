import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.sys.path.append('../')
import numpy as np
import soundkit.fit
import soundkit.dataprep
import tensorflow.keras.applications as imagenet_models

print('Load .wav')
specgen = soundkit.dataprep.spec.SpecGenerator(sample_seconds=1.0)
wav, sr = soundkit.dataprep.audio.load('example.wav', sr=specgen.sample_rate)
print('Audio shape:',wav.shape)
print('Audio seconds:',len(wav)/sr,'\n')

print('Load spectrogram')
spec = specgen.wav_to_spec(wav)
print('Spectrogram Shape:',spec.shape,'\n')

print('Get spectrogram model:')
model = soundkit.fit.models.ImageNetModel(backbone=imagenet_models.MobileNetV2, 
                                          num_classes=2,
                                          input_shape=spec.shape)
print('Prediction:',model.predict(spec[np.newaxis,...], verbose=0),'\n')

print('Get audio model:')
print('Sample seconds:',specgen.sample_seconds)
model.set_wav_input(True, specgen)
model.hop_seconds = 1.0
p = model.predict(wav[np.newaxis,...], verbose=0)
print('Prediction hop_seconds = '+str(model.hop_seconds)+':\n',p)

model.hop_seconds = 0.5
p = model.predict(wav[np.newaxis,...], verbose=0)
print('Prediction hop_seconds = '+str(model.hop_seconds)+':\n',p)


