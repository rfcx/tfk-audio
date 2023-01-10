import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from . import audio
import matplotlib.pyplot as plt


class SpecGenerator():
    ''' Tensorflow-compatible spectrogram data handler
    '''
    def __init__(self, 
                 sample_rate=16000,
                 stft_window_seconds=0.025,
                 stft_hop_seconds=0.01,
                 min_hz=None,
                 max_hz=None,
                 db_scale=True,
                 db_limits=(-100,20),
                 mel_bands=None,
                 tflite_compatible=True,
                 sample_seconds=3.0):
        '''
        Args:
            sample_rate:            target sample rate of processed audio
            stft_window_seconds:    seconds of audio processed in each STFT frame
            stft_hop_seconds:       seconds shifted between STFT frames
            min_hz:                 target min hz of computed spectrograms
            max_hz:                 target max hz of computed spectrograms
            db_scale:               whether to apply dB scaling to amplitudes
            db_limits:              dB values will be clipped within this range
            mel_bands:              number of mel bands to apply
            tflite_compatible:      if True will use a tflite-compatible STFT operation
            sample_seconds:         model input seconds (only used in fit module)
        '''
        
        self.sample_rate = sample_rate
        self.stft_window_seconds = stft_window_seconds
        self.stft_hop_seconds = stft_hop_seconds
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.db_scale = db_scale
        self.db_limits = db_limits
        self.sample_seconds = sample_seconds
        self.mel_bands = mel_bands
        self.second_width = int(1/self.stft_hop_seconds-(self.stft_window_seconds/self.stft_hop_seconds)+1)
        self.sample_width = None
        if self.sample_seconds is not None:
            self.sample_width = int((self.sample_seconds)/self.stft_hop_seconds-\
                                    (self.stft_window_seconds/self.stft_hop_seconds)+1)
        if self.max_hz is None:
            self.max_hz = self.sample_rate/2.0
        if self.min_hz is None:
            self.min_hz = 0.0
        self.stft_window_samples = int(self.stft_window_seconds*self.sample_rate)
        self.stft_hop_samples = int(round(self.sample_rate * self.stft_hop_seconds))
        self.fft_length = 2 ** int(np.ceil(np.log(self.stft_window_samples) / np.log(2.0)))
        if self.mel_bands is not None and self.mel_bands>0:
            self.num_frequency_bins = self.mel_bands
        else:
            self.num_frequency_bins = self.fft_length // 2 + 1
        self.image_shape = (self.sample_width, self.num_frequency_bins)
        self.tflite_compatible = tflite_compatible
        self._spec_file_sig = '_spec.npy'
        self._processed_files = set()

    def wav_to_spec(self, waveform):
        '''Converts a 1-D waveform into a spectrogram

        Args:
            waveform:               [<# samples>,]
            
        Returns:
            spec: [<# frequency bins>, <# time frames>]
        '''
        if type(waveform)==str:
            waveform, _ = audio.load_wav(waveform, sample_rate)
        return _wav_to_spec(waveform,
                            sample_rate = self.sample_rate,
                            stft_window_samples = self.stft_window_samples,
                            stft_hop_samples = self.stft_hop_samples,
                            min_hz = self.min_hz,
                            max_hz = self.max_hz,
                            fft_length = self.fft_length,
                            db_limits = self.db_limits,
                            db_scale = self.db_scale,
                            mel_bands = self.mel_bands,
                            tflite_compatible = self.tflite_compatible)
    
    def save_spec(self, audio_path):
        ''' Saves a spectrogram file
        '''
        spec = self.wav_to_spec(audio_path)
        np.save(audio_path+self._spec_file_sig, spec)
        
    def _save_spec(self, path, count, update=100):
        self.save_spec(path)
        if count%update==0:
            print(count)
        self._processed_files.add(path+self._spec_file_sig)
    
    def process_folder(self, folder, ext='.wav', overwrite=True, update=100, limit=None):
        '''Generates a spectrogram file for each audio file in a directory
        
        Args:
            folder: path to folder in which to search for audio files
            ext: extension of files to process
            overwrite: whether to overwrite previous spectrogram files
            update: interval of files processed print statement
            limit: limit of files to process (added for demo-ing)
        '''
        files_total = []
        files_to_process = []
        for root, dirs, files in os.walk(folder):
            for name in files:
                if name.endswith(ext):
                    files_total.append(os.path.join(root, name))
                    if not os.path.exists(os.path.join(root, name+self._spec_file_sig)):
                        files_to_process.append(os.path.join(root, name))
        if overwrite:
            files_to_process = files_total
        for i in list(set(files_total).difference(files_to_process)):
            self._processed_files.add(i+self._spec_file_sig)
        print('Audio files found:',len(files_total))
        print('Spectrogram files found:',len(files_total)-len(files_to_process))
        print('To process:',len(files_to_process))
        for c,i in enumerate(files_to_process[:limit]):
            self._save_spec(i, c, update=update)         
    
    def plot_example(self, x=None, dblims=list([-100, 20])):
        ''' Plots an example spectrogram
        
        Args:
            x: one of
                path to spectrogram file
                path to audio file
                waveform array
                spectrogram array
                None (will look for a random example in _processed_files attribute)
            dblims: dB range of output (assumes dB scaling is applied by default)
        '''
        if (x is None):
            assert len(self._processed_files)>0, 'Error: No files found.'
            tmp = list(self._processed_files)
            random.shuffle(tmp)
            tmp = tmp[0]
        else:
            tmp = x
        if isinstance(tmp, str):
            if tmp.endswith(self._spec_file_sig):
                spec = np.load(tmp)
            else:
                try:
                    wav, sr = audio.load_wav(tmp, self.sample_rate)
                    spec = self.wav_to_spec(wav)
                except Exception as e:
                    print(e)
                    print('Error: Could not interpret input as path to audio or spectrogram file')
        elif (isinstance(tmp, tf.Tensor)) or (isinstance(tmp, np.ndarray)):
            assert len(np.shape(tmp)) in (1,2), 'Error: Array inputs are expected to have 1 (waveform) or 2 (spectrogram) dimensions'
            if len(np.shape(tmp))==1:
                spec = self.wav_to_spec(tmp)
            elif len(np.shape(tmp))==2:
                spec = tmp
        spec = tf.transpose(spec) # [<# frequency bands>, <# time frames>]

        plt.figure(figsize=(5,4))
        plt.pcolormesh(spec);
        for i in range(2):
            if not self.db_limits[i] is None:
                dblims[i] = self.db_limits[i]
        plt.clim(dblims);
        plt.axis('off')
        plt.colorbar(aspect=20, label='dB');

    def plot_examples(self, path=None, dblims=list([-100, 20])):
        ''' Plots a grid of example spectrograms
        
        Args:
            path: path to folder in which to search for spectrogram files
            dblims: dB range of output (assumes dB scaling is applied by default)
        '''
        if (path is None):
            assert len(self._processed_files)>0, 'Error: No spectrogram files given or processed with process_folder.'
            tmp = list(self._processed_files)
        else:
            if not path.endswith('/'):
                path+='/'
            tmp = [path+i for i in os.listdir(path) if i.endswith(self._spec_file_sig)]
        assert len(tmp)>0, 'Error: No spectrogram files found.'
        random.shuffle(tmp)
        plt.figure(figsize=(10,10))
        nr=4
        nc=4
        for c,i in enumerate(list(tmp)[:(nr*nc)]):
            plt.subplot(nr,nc,c+1)
            spec = np.load(i)
            spec = tf.transpose(spec) # [<# frequency bands>, <# time frames>]
            plt.pcolormesh(spec);
            for i in range(2):
                if not self.db_limits[i] is None:
                    dblims[i] = self.db_limits[i]
            plt.clim(dblims);
            plt.axis('off')
            
    def shape(self, input_seconds):
        ''' Spectrogram shape for a given waveform duration
        '''
        width = int((input_seconds)/self.stft_hop_seconds-\
                    (self.stft_window_seconds/self.stft_hop_seconds)+1)
        return (width, self.num_frequency_bins)
    
    def to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump({k:v for (k,v) in self.__dict__.items() if k!='_processed_files'}, f, sort_keys=True, indent=4)
    
    def from_json(self, filename):
        config = json.load(open(filename, 'r'))
        for key in config:
            setattr(self, key, config[key])
        self._processed_files = set()
        
#     def _process_batch(self, batch):
#         ''' Attempt at allowing >1 batch size for audio models
#
#             One option is to predefine the input waveform length and pad everything to it
#         '''
#         return tf.map_fn(lambda x: self.wav_to_spec(x), elems=(batch))

    
def _wav_to_spec(waveform, 
                 sample_rate, 
                 stft_window_samples, 
                 stft_hop_samples, 
                 min_hz,
                 max_hz,
                 fft_length, 
                 db_scale,
                 db_limits,
                 mel_bands,
                 tflite_compatible):
    '''Converts a 1-D waveform into a spectrogram
    
    Separating this function from the SpecGenerator class makes it serializable for audio model layers

    Args:
        waveform:               [<# samples>,]
        sample_rate:            target sample rate of processed audio
        stft_window_seconds:    seconds of audio processed in each STFT frame
        stft_hop_seconds:       seconds shifted between STFT frames
        min_hz:                 target min hz of computed spectrograms
        max_hz:                 target max hz of computed spectrograms
        fft_length:             number of Fourier coefficients to compute
        db_scale:               whether to apply dB scaling to amplitudes
        db_limits:              dB values will be clipped within this range
        mel_bands:              number of mel bands to apply
        tflite_compatible:      if True will use a tflite-compatible STFT operation
    Returns:
        spec: [<# frequency bins>, <# time frames>]
    '''
    if tflite_compatible:
        spec = _tflite_stft_magnitude(
            signal=waveform,
            frame_length=stft_window_samples,
            frame_step=stft_hop_samples,
            fft_length=fft_length)
    else:
        spec = tf.abs(tf.signal.stft(
            signal=waveform,
            frame_length=stft_window_samples,
            frame_step=stft_hop_samples,
            fft_length=fft_length))
    if mel_bands is not None and mel_bands>0:
        spec = _apply_mel_scale(spec,
                                mel_bands,
                                sample_rate,
                                min_hz,
                                max_hz)
    else:
        spec = _crop(spec, 
                     sample_rate, 
                     fft_length // 2 + 1,
                     min_hz,
                     max_hz)
    if db_scale:
        spec = _apply_db_scale(spec, db_limits)
    return spec
    # [<# time frames>, <# frequency bands>]

        
def _crop(spec, sample_rate, num_frequency_bins, min_hz, max_hz):
    '''Crop a spectrogram to frequency range

    Args:
        spec: [<# time frames>, <# frequency bands>]
    '''
    linear_frequencies = math_ops.linspace(0.0, sample_rate/2.0, num_frequency_bins)
    idx_min = tf.squeeze(tf.where(linear_frequencies>=min_hz))[0]
    idx_max = tf.squeeze(tf.where(linear_frequencies<=max_hz))[-1]+1
    return spec[...,idx_min:idx_max]


def _apply_mel_scale(spec, mel_bands, sample_rate, min_hz, max_hz):
    '''Converts a linear spectrogram into a mel-scaled spectrogram
    Args:
        spec: array with shape [<# time frames>, <# frequency bands>]
    Returns:
        mel_spectrogram: spectrogram with shape [<# time frames>, <# frequency bands>]
    '''
    # Convert spectrogram into mel spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_bands,
        num_spectrogram_bins=spec.shape[1],
        sample_rate=sample_rate,
        lower_edge_hertz=min_hz,
        upper_edge_hertz=max_hz)
    mel_spectrogram = tf.matmul(spec, linear_to_mel_weight_matrix)
    return mel_spectrogram


def _apply_db_scale(spec, db_limits=[None, None]):
    ''' Apply db scaling to spectrogram amplitudes
    '''
    spec = 10*tf.math.log(spec)
    if db_limits[0] is None:
        db_limits = (tf.math.reduce_min(spec), db_limits[1])
    if db_limits[1] is None:
        db_limits = (db_limits[0], tf.math.reduce_max(spec))    
    return tf.clip_by_value(spec, db_limits[0], db_limits[1])

            
def _tflite_stft_magnitude(signal, frame_length, frame_step, fft_length):
    '''TF-Lite-compatible version of tf.abs(tf.signal.stft()).
    
    Taken from https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/features.py
    
    '''
    def _hann_window():
        return tf.reshape(
            tf.constant(
                (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
                ).astype(np.float32),
                name='hann_window'), [1, frame_length])

    def _dft_matrix(dft_length):
        '''Calculate the full DFT matrix in NumPy.'''
        # See https://en.wikipedia.org/wiki/DFT_matrix
        omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
        # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
        return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

    def _rdft(framed_signal, fft_length):
        '''Implement real-input Discrete Fourier Transform by matmul.'''
        # We are right-multiplying by the DFT matrix, and we are keeping only the
        # first half ('positive frequencies').  So discard the second half of rows,
        # but transpose the array for right-multiplication.  The DFT matrix is
        # symmetric, so we could have done it more directly, but this reflects our
        # intention better.
        complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(fft_length // 2 + 1), :].transpose()
        real_dft_matrix = tf.constant(
            np.real(complex_dft_matrix_kept_values).astype(np.float32),
            name='real_dft_matrix')
        imag_dft_matrix = tf.constant(
            np.imag(complex_dft_matrix_kept_values).astype(np.float32),
            name='imaginary_dft_matrix')
        signal_frame_length = tf.shape(framed_signal)[-1]
        half_pad = (fft_length - signal_frame_length) // 2
        padded_frames = tf.pad(
            framed_signal,
            [
                # Don't add any padding in the frame dimension.
                [0, 0],
                # Pad before and after the signal within each frame.
                [half_pad, fft_length - signal_frame_length - half_pad]
            ],
            mode='CONSTANT',
            constant_values=0.0)
        real_stft = tf.matmul(padded_frames, real_dft_matrix)
        imag_stft = tf.matmul(padded_frames, imag_dft_matrix)
        return real_stft, imag_stft

    def _complex_abs(real, imag):
        return tf.sqrt(tf.add(real * real, imag * imag))

    framed_signal = tf.signal.frame(signal, frame_length, frame_step)
    windowed_signal = framed_signal * _hann_window()
    real_stft, imag_stft = _rdft(windowed_signal, fft_length)
    stft_magnitude = _complex_abs(real_stft, imag_stft)
    return stft_magnitude


