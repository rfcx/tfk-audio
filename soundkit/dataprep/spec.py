import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
import soundkit.dataprep.audio
from soundkit.config import get_config
import matplotlib.pyplot as plt

class SpecGenerator():
    def __init__(self, 
                 sample_rate=16000,
                 stft_window_seconds=0.025,
                 stft_hop_seconds=0.01,
                 db_scale=True,
                 min_hz=None,
                 max_hz=None,
                 sample_seconds=3.0,
                 db_limits=(None,None),
                 mel_bands=None,
                 tflite_compatible=True):
        
        self.sample_rate = sample_rate
        self.stft_window_seconds = stft_window_seconds
        self.stft_hop_seconds = stft_hop_seconds
        self.db_scale = db_scale
        self.db_limits = db_limits
        self.min_hz = min_hz
        self.max_hz = max_hz
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
        self.num_frequency_bins = self.fft_length // 2 + 1
        self.tflite_compatible = tflite_compatible
        self._processed_files = set()
        
    def wav_to_spec(self, waveform):
        """Converts a 1-D waveform into a spectrogram
        """
        if type(waveform)==str:
            waveform, _ = soundkit.dataprep.audio.load(waveform, self.sample_rate)
        if self.tflite_compatible:
            spec = _tflite_stft_magnitude(
                signal=waveform,
                frame_length=self.stft_window_samples,
                frame_step=self.stft_hop_samples,
                fft_length=self.fft_length)
        else:
            spec = tf.abs(tf.signal.stft(
                signal=waveform,
                frame_length=self.stft_window_samples,
                frame_step=self.stft_hop_samples,
                fft_length=self.fft_length))
        if self.mel_bands is not None and self.mel_bands>0:
            spec = self.apply_mel_scale(spec)
        else:
            spec = self.crop(spec)
        if self.db_scale:
            spec = self.apply_db_scale(spec, self.db_limits)
        spec = tf.transpose(spec) # [<# frequency bands>, <# time frames>]
            
        return spec
    
    def save_spec(self, audio_path):
        spec = self.wav_to_spec(audio_path)
        np.save(audio_path+'_spec.npy', spec)
        
    def _save_spec(self, path, count, update=100):
        self.save_spec(path)
        if count%update==0:
            print(count)
        self._processed_files.add(path+'_spec.npy')
    
    def crop(self, spec):
        """Crop a spectrogram a desired frequency range
        """
        linear_frequencies = math_ops.linspace(0.0, self.sample_rate/2.0, self.num_frequency_bins)
        idx_min = tf.squeeze(tf.where(linear_frequencies>=self.min_hz))[0]
        idx_max = tf.squeeze(tf.where(linear_frequencies<=self.max_hz))[-1]+1
        return spec[...,idx_min:idx_max]
    
    def apply_mel_scale(self, spec):
        """Converts a linear spectrogram into a mel-scaled spectrogram
        Args:
            spec: array with shape [<# frequency bands>, <# time frames>]
        Returns:
            mel_spectrogram: spectrogram with shape [<# frequency bands>, <# time frames>]
        """
        # Convert spectrogram into mel spectrogram.
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.mel_bands,
            num_spectrogram_bins=spec.shape[0],
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.min_hz,
            upper_edge_hertz=self.max_hz)
        spec = tf.transpose(spec)
        mel_spectrogram = tf.matmul(spec, linear_to_mel_weight_matrix)
        mel_spectrogram = tf.transpose(mel_spectrogram)

        return mel_spectrogram
    
    def apply_db_scale(self, spec, db_limits=[None, None]):
        """ Apply db scaling to spectrogram amplitudes
        """
        spec = 10*tf.math.log(spec)
        if db_limits[0] is None:
            db_limits = (tf.math.reduce_min(spec), db_limits[1])
        if db_limits[1] is None:
            db_limits = (db_limits[0], tf.math.reduce_max(spec))
            
        return tf.clip_by_value(spec, db_limits[0], db_limits[1])
    
    def process_folder(self, folder, ext='.wav', update=100, limit=None):
        """Generates a spectrogram file for each audio file in a directory"""
        files_to_process = []
        for root, dirs, files in os.walk(folder):
            for name in files:
                if name.endswith(ext):
                    files_to_process.append(os.path.join(root, name))
        print('Files found:',len(files_to_process))
        for c,i in enumerate(files_to_process[:limit]):
            self._save_spec(i, c)         
    
    def plot_example(self, file=None, dblims=list([-150, 20])):
        if (file is None):
            assert len(self._processed_files)>0, "Error: No files found."
            tmp = list(self._processed_files)
            random.shuffle(tmp)
            tmp = tmp[0]
        else:
            tmp = file
        plt.figure(figsize=(5,4))
        spec = np.load(tmp)
        plt.pcolormesh(spec);
        for i in range(2):
            if not self.db_limits[i] is None:
                dblims[i] = self.db_limits[i]
        plt.clim(dblims);
        plt.axis('off')
        plt.colorbar(aspect=20, label='dB');

    def plot_examples(self, path=None, dblims=list([-150, 20])):
        if (path is None):
            assert len(self._processed_files)>0, "Error: No files found."
            tmp = list(self._processed_files)
        else:
            tmp = [path+i for i in os.listdir(path) if i.endswith('_spec.npy')]
        assert len(tmp)>0, "Error: No files found."
        random.shuffle(tmp)
        plt.figure(figsize=(10,10))
        nr=4
        nc=4
        for c,i in enumerate(list(tmp)[:(nr*nc)]):
            plt.subplot(nr,nc,c+1)
            spec = np.load(i)
            plt.pcolormesh(spec);
            for i in range(2):
                if not self.db_limits[i] is None:
                    dblims[i] = self.db_limits[i]
            plt.clim(dblims);
            plt.axis('off')

            
def _tflite_stft_magnitude(signal, frame_length, frame_step, fft_length):
    """TF-Lite-compatible version of tf.abs(tf.signal.stft())."""
    def _hann_window():
        return tf.reshape(
            tf.constant(
                (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
                ).astype(np.float32),
                name='hann_window'), [1, frame_length])

    def _dft_matrix(dft_length):
        """Calculate the full DFT matrix in NumPy."""
        # See https://en.wikipedia.org/wiki/DFT_matrix
        omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
        # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
        return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

    def _rdft(framed_signal, fft_length):
        """Implement real-input Discrete Fourier Transform by matmul."""
        # We are right-multiplying by the DFT matrix, and we are keeping only the
        # first half ("positive frequencies").  So discard the second half of rows,
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


