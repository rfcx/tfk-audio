import os
import librosa
import tensorflow as tf
from ffmpy import FFmpeg, FFRuntimeError


def load_wav(path, target_sr=None, numpy=False, print_audio_info=False):
    '''Loads a 16-bit PCM .wav file into a tensor
    
    Args:
        path: .wav file path
        target_sr: desired sample rate of loaded audio
        numpy: whether to return loaded audio as numpy array
        print_audio_info: whether to print audio info (duration, sample rate)
    
    Returns:
        y: time series of amplitudes with shape [<# samples>,]
        sr: sample rate of the recording
        
    '''

    if target_sr is not None:
        target_sr = int(target_sr)

    y, sr = tf.audio.decode_wav(tf.io.read_file(path)) # load with original sample rate
    if len(tf.shape(y))>1:
        y = tf.reduce_mean(y, axis=1)
    if target_sr is not None and target_sr != sr: # maybe resample
        y = resample(y.numpy(), sr.numpy(), target_sr)
        y = tf.convert_to_tensor(y)
        sr = target_sr
    if print_audio_info:
        print(f'Duration: {y.shape[0] / sr:.2f} seconds')
        print(f'Sample rate: {sr} Hz')
    if numpy:
        return y.numpy(), sr.numpy()
    else:
        return y, sr


def resample(wav, sr, sr_new):
    '''Resamples a waveform
    '''
    return librosa.resample(wav, orig_sr=sr, target_sr=sr_new)


def convert_to_wav(source, destination, codec='pcm_s16le', sample_rate='64000'):
    ''' Converts a file to WAV format
    '''
    ff = FFmpeg(global_options='-loglevel panic',
                inputs={source: None},
                outputs={
                    destination:
                    f'-flags +bitexact -y -acodec {codec} -ar {sample_rate} -ac 1 -ss 0 -f wav'
                })
    try:
        result = ff.run()
    except FFRuntimeError as e:
        print(f'convert_to_wav debugging source file size: {os.path.getsize(source)}')
        print(f'convert_to_wav debugging stdout: {e.stdout}')
        stderr_compact = e.stderr.replace('\n', ' ').replace('\r', ' ') if e.stderr else 'None'
        result = None, f'FFRuntimeError: {e.exit_code} {stderr_compact}'
    del ff
    return result