soundkit (open to better name ideas)
================
A Python module for streamlining audio recognition

Submodules
----------------

dataprep
~~~~~~~~~~~~~
Audio and spectrogram handling, e.g.::

    from soundkit import dataprep
    
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
Audio data augmentation, e.g.::

    from soundkit.fit import datagen
    
    traingen = datagen.TrainGenerator(files_train,
                                      y_train,
                                      labels,
                                      batch_size = 16,
                                      augment = True,
                                      augment_blend_prob = 1,
                                      augment_max_time_masks = 5,
                                      augment_max_time_mask_size = 0.05,
                                      augment_add_noise_prob = 0,
                                      augment_max_time_shift = 0.5,
                                      )
    batch = traingen.__getitem__(0)
    datagen.plot_batch_samples(batch[0])
    
Models compatible with audio and spectrogram data, e.g.::

    from soundkit.fit import models
    
    # create model
    model = models.ImageNetModel(backbone=imagenet_models.MobileNetV2,
                                 num_classes=2,
                                 input_shape=spec.shape)
    # predict on spectrogram                             
    model.predict(spec[np.newaxis,...])
    
    # predict on audio
    model.set_wav_input(True, specgen)
    model.predict(wav[np.newaxis,...])
    
    # fit
    model.fit(traingen)
    
    
