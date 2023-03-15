# MoChA Speech Recognition

This repository contains an implementation of a speech recognition system using the Monotonic Chunkwise Attention (MoChA) mechanism. The goal of this project is to provide an end-to-end automatic speech recognition (ASR) system that can transcribe speech into text.

## Table of Contents

- [Installation](https://chat.openai.com/chat?model=gpt-4#installation)
- [Data Preparation](https://chat.openai.com/chat?model=gpt-4#data-preparation)
- [Training](https://chat.openai.com/chat?model=gpt-4#training)
- [Evaluation](https://chat.openai.com/chat?model=gpt-4#evaluation)
- [Usage](https://chat.openai.com/chat?model=gpt-4#usage)
- [License](https://chat.openai.com/chat?model=gpt-4#license)

## Installation

Before getting started, make sure to have the following prerequisites installed:

- Python 3.9
- TensorFlow 2
- Numpy
- Librosa
- tqdm

To install the required packages, you can run:

```
pip install -r requirements.txt
```

## Data Preparation

1. Download your desired speech recognition dataset (e.g., LibriSpeech, CommonVoice) and organize it into the `data/raw_data` directory, with separate subdirectories for train, dev, and test sets.
2. Run the preprocessing script to convert the raw audio files into suitable features (e.g., MFCCs, log-mel filterbank energies) and store them in the `data/preprocessed_data` directory:

```
python src/data_utils.py
```

## Training

To train the MoChA ASR model, run the following command:

```
python src/train.py
```

This will train the model using the preprocessed data and save the best performing model in the `models/saved_models` directory.

## Evaluation

To evaluate the trained model on the test dataset, run:

```
python src/evaluate.py
```

This will load the best performing model from `models/saved_models` and calculate the performance metrics (e.g., Word Error Rate) on the test dataset.

## Usage

After training and evaluating the model, you can use it for your own speech recognition tasks by importing the `MoChAASR` class from the `model.py` file and loading the trained weights.

```python
from src.model import MoChAASR

model = MoChAASR()
model.load_weights('path/to/saved/model/weights')
transcription = model.transcribe('path/to/audio/file')
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://chat.openai.com/LICENSE) file for more details.