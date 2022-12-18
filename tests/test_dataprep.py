import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.sys.path.append('../')
from tfk_audio import dataprep

# create a spectrogram generator
specgen = dataprep.spec.SpecGenerator(sample_rate=16000,
                                          stft_window_seconds=0.05,
                                          stft_hop_seconds=0.01,
                                          db_limits=(None, None))

# process a single file
wav, sr = dataprep.audio.load('data/example1.wav')
spec = specgen.wav_to_spec(wav)

# process all .wav files in a folder
specgen.process_folder('data', limit=None)

# check examples
specgen.plot_examples(path='data')
