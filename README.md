tfk-audio package - v1.0.2 
================
**T**ensor **F**low **K**eras Audio

**TFK-Audio** is a TensorFlow-based library for audio preprocessing and model training. It provides utilities for converting audio waveforms into spectrograms, creating dataloaders, data augmentation, and integrating audio data with deep learning models. The library is modular, with components for preprocessing (`audio` and `spec`) and model training (`datagen`, `layers`, `labels`, and `metrics`).

---

## Table of Contents
- [Installation](#installation)
- [Modules](#modules)
  - [Preprocessing](#preprocessing)
    - [Audio](#audio)
    - [Spec](#spec)
  - [Model Training](#model-training)
    - [Datagen](#datagen)
    - [Layers](#layers)
    - [Labels and Metrics](#labels-and-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/rfcx/tfk-audio.git
cd tfk-audio
pip install -r requirements.txt
pip install -e .
```

---

## Modules

### Preprocessing

#### Audio
`audio` provides functions for loading and processing audio files.
```python
# load a single file, and check duration and sample rate
wav, sr = audio.load_wav("example.wav", print_audio_info=True)
```

#### Spec
`spec` provides a SpecGenerator class for converting waveforms into spectrograms and managing spectrogram parameters throughout the workflow.
```python
# create a spectrogram generator
specgen = spec.SpecGenerator(
    sample_rate=sr,
    stft_window_seconds=0.05,
    stft_hop_seconds=0.01,
    db_limits=(None, None),
    sample_seconds=2.0
)
spec = specgen.wav_to_spec(wav)
# process all .wav files in a folder
specgen.process_folder(indir="data/wav", outdir="data/spectrogram", limit=None)
# check examples for a specific class
specgen.plot_examples(path="data/spectrogram/chainsaw")
# save and load spectrogram generator parameters
specgen.to_json("data/demo_specgen.json")
specgen.from_json("data/demo_specgen.json")
```

---

### Model Training

#### Datagen
`datagen` provides utilities for efficient spectrogram data loading and augmentation using TensorFlow datasets.
```python
# get list of spectrogram file paths and label map
# also supports creating dataset partitions and resampling classes
files_train, _, _, label_map = datagen.get_files_and_label_map("data/spectrogram/")
# save the spectrograms and labels as TFRecord files
tfrecord_files = datagen.create_tfrecords(
    files=files_train,
    labels=labels_train,
    outdir="data/tfrecords/",
    batch_size=10,
    overwrite=True
)
# create a TF Dataset with augmentation
dataset = datagen.spectrogram_dataset_from_tfrecords(
    files=tfrecord_files,
    image_shape=specgen.image_shape,
    nclass=len(label_map),
    batch_size=10,
    augment=True,  # Enable data augmentation
    label_smoothing=0.1  # Apply label smoothing
)
```

#### Layers
`layers` provides TensorFlow-Keras compatible layers for working with spectrogram and waveform inputs and common image classification models.
```python
# choose waveform or spectrogram input
# recommended approach is to train using spectrogram input and 
# change the model input layer for deployment on audio waveform data
use_waveform = False  # Set to False if you want to use spectrogram inputs
if use_waveform:
    preprocess_layer = WavImageNet(target_shape=(224, 224), spec_params=specgen.get_config())
    input_shape = (int(specgen.sample_rate * specgen.sample_rate),)
else:
    preprocess_layer = SpecImageNet(target_shape=(224, 224))
    input_shape = (None, None, 1)  # Spectrogram input shape
# load a pre-trained ResNet50 model (without the top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# add the preprocessing layer as the head
inputs = tf.keras.Input(shape=input_shape)
x = preprocess_layer(inputs)
x = base_model(x)
x = GlobalAveragePooling2D()(x)
outputs = Dense(len(label_map), activation='softmax')(x)  # Example: 10 output classes
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, None, None, 1)]   0         
                                                                 
#  spec_to_imagenet (Function  (None, 224, 224, 3)       0         
#  al)                                                             
                                                                 
#  resnet50 (Functional)       (None, 7, 7, 2048)        23587712  
                                                                 
#  global_average_pooling2d_2  (None, 2048)              0         
#   (GlobalAveragePooling2D)                                       
                                                                 
#  dense_2 (Dense)             (None, 2)                 4098      
                                                                 
# =================================================================
# Total params: 23591810 (90.00 MB)
# Trainable params: 23538690 (89.79 MB)
# Non-trainable params: 53120 (207.50 KB)
# _________________________________________________________________
```

#### Labels and Metrics
`labels` and `metrics` provide tools for preparing label arrays of various formats. Supports weak-labeling and masked loss functions.
```python
# get labels based on file list
# supports various label formats
labels_train = labels.get_classification_labels(
    files=files_train, 
    label_map=label_map,
    label_format='multi-label'
)
# wrap a loss function to handle masked (unknown) elements
masked_loss = metrics.MaskedLoss(tf.keras.losses.BinaryCrossentropy(), mask_val=-1.0)
```
---

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

---

## License
This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for details.
