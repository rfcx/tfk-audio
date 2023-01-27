import tensorflow as tf


def predict_wav(wav, audio_model, specgen, hop_seconds=1.0):
    ''' Runs prediction on a waveform
    
    Args
        wav:            1D waveform
        audio_model:    model that takes a waveform as input
        specgen:        SpecGenerator of the given model
        hop_seconds:    seconds between prediction start times
        
    Returns
        p:              model predictions
    '''
    wavs = []
    
    # number of seconds in a sample
    sample_length = int(specgen.sample_rate*specgen.sample_seconds)
    
    for i in range(0, len(wav), int(specgen.sample_rate*hop_seconds)):
        
        # get current window
        window = wav[i:(i+sample_length)]
        
        # pad if needed
        if not len(window)==sample_length:
            window = tf.pad(window, [[0, sample_length-len(window)]])
        
        wavs.append(window)
    
    # stack
    wavs = tf.stack(wavs)
    
    # predict
    p = audio_model.predict(wavs, verbose=0)
    
    return p

