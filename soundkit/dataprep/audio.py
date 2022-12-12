import os
import librosa
from ffmpy import FFmpeg, FFRuntimeError


def load(path, sr=None):
    """Loads an audio file
    
    Args:
        path: audio file path
    
    Returns:
        y: time series of amplitudes with shape [<# samples>,]
        sr: sample rate of the recording
        
    """
    y, sr0 = librosa.load(path, sr=None)
    if sr is not None:
        y = resample(y, sr0, sr)
    return y, sr


def resample(wav, sr, sr_new):
    """Resamples a waveform
    """
    return librosa.resample(wav, orig_sr=sr, target_sr=sr_new)


def convert_to_wav(source, destination, codec='pcm_s24le', sample_rate='64000'):
    """ Converts a file to WAV format
    """
    ff = FFmpeg(global_options='-loglevel panic',
                inputs={source: None},
                outputs={
                    destination:
                    f"-flags +bitexact -y -acodec {codec} -ar {sample_rate} -ac 1 -ss 0 -f wav"
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