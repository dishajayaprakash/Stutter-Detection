# Classifying Stuttering Events

Jump to section:

- [Model details](#model-details)
- [Intended use](#intended-use)
- [Factors](#factors)
- [Metrics](#metrics)
- [Training data](#training-data)
- [Ethical considerations](#ethical-considerations)

## Model details

_Basic information about the model._

Review section 4.1 of the [model cards paper](https://arxiv.org/abs/1810.03993).

This model was developed by Sean Foley and Disha Thotappala Jayaprakash from USC in the Fall of 2023. The objective of this model is to improve the classification of stuttering events when given short audio clips of speech. Specifically, the model was trained to classify 3-second audio clips into either a binary label of fluent/disfluent speech or into one of five categories 'Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', or 'NoStutteredWords'. 

Three architectures were employed for these two tasks: LSTM, ConvLSTM, and ResNet. 

The following is the list of training algorithms and parameters:

1. Losses: binary and multi-class cross-entropy
2. Learning rates: 1e2, 1e3, 1e4
3. Batch sizes: 16, 32, 64, 128
4. Optimizer: Adam
5. Regularization: L2

The following features were input to the model:

1. Mel-filter banks (MFB)/Spectrogram: MFBs are a sequence of filters used to identify the power spectrum of sound based on the mel scale, which is a perceptual scale of pitches. A spectrogram is a visual representation of the spectrum of frequencies in a sound or signal as they vary with time. We extracted 40-dimensional MFBs, with frequency cut-offs at 0Hz and 8000Hz.
2. Fundamental frequency (F0): F0, or pitch, is the lowest frequency in a sound and is a key characteristic in speech and music. The pitch-delta refers to changes in the pitch over time, providing dynamic information about the sound. Voicing features are also included, which indicate which pitch points are from voiced sounds. 
3. Wav2vec 2.0 representations (Baevski et al. 2020) : Latent representations extracted from a single transformer layer after feeding the audio clips to the pretrained model. The input gets passed through 7 convolutional layers and 12 transformer blocks, with the weights being frozen. Features were extracted from the 7th transformer layer, which has been shown to encode meaningful acoustic information (Pasad et al. 2023). The resulting matrix of features was averaged over the hidden units (d=768), resulting in a 1D vector of encodings. 

The code for the model can be found [here](https://github.com/sfoley3/sep28k_stuttering_detection). 

## Intended use

The primary intended use of the models is to classify stuttering events given short audio clips of speech. The goal is to improvide indentification and classification of stuttering so that speech technology can be improved for this and other forms of atypical speech, making such technology more equitable. 

## Factors

Information regarding demographics of the speakers included in the applied corpus were not indicated by the original authors. The data was gathered from a small set of podcasts containing stuttered speech. 

## Metrics

F1 was used to assess the performance of the models. Two F1 scores were computed for each model:

1. Binary F1
2. Multiclass F1

### Model performance measures

Model performance was assessed by its binary and multiclass F1 scores when applied to the held-out test set. 

### Datasets

The Sep28K corpus was used in training. This corpus was released by Lea et al. (2021) and contained 28K 3-second audio clips of stuttering events scraped from publicly available podcasts. In addition to the podcast data, the FluencyBank corpus was also included in this dataset. This corpus contains longform conversational data from multiple subjects with developmental stuttering. After processing the available files, a total of 20,480 audio clips were used in training the models. 

### Motivation

The motivation is to improve identification and classification of stuttering so that speech technology can be improved for this and other forms of atypical speech, making such technology more equitable. 

### Preprocessing

The 20,480 audio clips were preprocessed using the features described above, resulting in a dictionary of features containing three feature sets for each file. 

## Training data

The data was split into training, validation and test subsets, using a split of (0.6, 0.2, 0.2) respectively. 

## Ethical considerations

The corpus used in training is not meant to be inclusive of all forms of stuttering, given that stuttering is highly variable in how it is manifested. As such, caution should be used in generalizing the results to stuttering more broadly. This likely also limits the ability of the models to be extended to unseen data from other corpora containing different speakers. 

