tfk-audio
================
**T**\ ensor\ **F**\ low **K**\ eras Audio

Submodules
----------------

dataprep
~~~~~~~~~~~~~
Audio and spectrogram handling, e.g.::

    from tfk_audio import dataprep
    
    # load audio
    wav, sr = dataprep.audio.load('example.wav')
    
    # load spectrogram
    specgen = dataprep.spec.SpecGenerator(sample_rate = sr,
                                          stft_window_seconds = 0.05)
    spec = specgen.wav_to_spec('example.wav')
    spec = specgen.wav_to_spec(wav)
    
    # create a spectrogram file for each .wav in a folder
    specgen.process_folder('./tmp/')
    
    # plot
    specgen.plot_examples()


fit
~~~~~~~~~~~~~
To fill
    
    
