# Classifying Stuttering Events with Sep28K

This repository contains code for preprocessing the [Sep28k](https://github.com/apple/ml-stuttering-events-dataset) corpus and training a model for both binary (fluency/disfluency) and multiclass ('Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords') classification. 

The user is assumed to have downloaded the relevant files from the original Sep28k repo. 

Install required libraries:

```pip install requirements.txt```

The first step is to run the following:

```python preprocess.py```

This returns a dictionary containing F0, MFB, and wav2vec 2.0 features, and dictionaries for the labels and audio file paths. 

To train a model on these features:

```python train.py --model --batch_size --num_epochs```

This `train.py` allows one to select a model from the `models.py` file and generates the dataset from the `dataset.py` file.

Available models include: ConvLSTM(), LSTM_base(), ResNet()

Following training the `utils.py` contains a plotting function to show the binary and multiclass losses and F1 scores. 
