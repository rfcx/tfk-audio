import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.sys.path.append('../')
import soundkit.dataprep

print('Generate spectrogram')
specgen = soundkit.dataprep.spec.SpecGenerator(sample_seconds=1.0)
wav, sr = soundkit.dataprep.audio.load('example.wav', sr=specgen.sample_rate)
print('Audio shape:',wav.shape)
print('Audio seconds:',len(wav)/sr)
spec = specgen.wav_to_spec(wav)
print('Spectrogram Shape:',spec.shape,'\n')
