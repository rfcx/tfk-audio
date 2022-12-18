import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.sys.path.append('../')
import tf_ark as ark

specgen = ark.dataprep.spec.SpecGenerator(sample_rate=16000,
                                          stft_window_seconds=0.05,
                                          stft_hop_seconds=0.01,
                                          db_limits=(None, None))

# process a single file
wav, sr = ark.dataprep.audio.load('data/example1.wav')
spec = specgen.wav_to_spec(wav)

# process all .wav files in a folder
specgen.process_folder('data', limit=None)

# check examples
specgen.plot_examples(path='data')
