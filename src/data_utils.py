import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T

    # Standardize MFCC features
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc)

    return mfcc


def process_audio_files(src_path, dst_path, dataset_type):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                mfcc = extract_mfcc(file_path)
                file_id = file[:-5]
                speaker_id = root.split(os.path.sep)[-2]
                dest_folder = os.path.join(dst_path, speaker_id)
                os.makedirs(dest_folder, exist_ok=True)
                np.save(os.path.join(dest_folder, file_id + '.npy'), mfcc)


def process_transcripts(src_path, dst_path, dataset_type):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                speaker_id = root.split(os.path.sep)[-2]
                dest_folder = os.path.join(dst_path, speaker_id)
                os.makedirs(dest_folder, exist_ok=True)
                with open(file_path, 'r') as transcript_file:
                    lines = transcript_file.readlines()
                    for line in lines:
                        file_id, text = line.strip().split(' ', 1)
                        dest_file_path = os.path.join(dest_folder, file_id + '.txt')
                        with open(dest_file_path, 'w') as dest_file:
                            dest_file.write(text)


def preprocess_librispeech(src_path, dst_path):
    for dataset_type in ['train', 'dev', 'test']:
        process_audio_files(os.path.join(src_path, dataset_type),
                            os.path.join(dst_path, dataset_type),
                            dataset_type)
        process_transcripts(os.path.join(src_path, dataset_type),
                            os.path.join(dst_path, dataset_type),
                            dataset_type)


def load_data(data_path):
    train_data, dev_data, test_data = [], [], []
    for dataset_type in ['train', 'dev', 'test']:
        audio_data = []
        transcript_data = []
        dataset_path = os.path.join(data_path, dataset_type)
        for speaker_id in os.listdir(dataset_path):
            speaker_path = os.path.join(dataset_path, speaker_id)
            for file in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file)
                if file.endswith('.npy'):
                    audio_data.append(np.load(file_path))
                elif file.endswith('.txt'):
                    with open(file_path, 'r') as transcript_file:
                        transcript_data.append(transcript_file.read().strip())

        if dataset_type == 'train':
            train_data = (audio_data, transcript_data)
        elif dataset_type == 'dev':
            dev_data = (audio_data, transcript_data)
        elif dataset_type == 'test':
            test_data = (audio_data, transcript_data)

    return train_data, dev_data, test_data


def pad_sequence(sequences, max_length, padding_value=0):
    return np.array(
        [np.pad(seq, (0, max_length - len(seq)), mode='constant', constant_values=padding_value) for seq in sequences])


def create_dataset(data, batch_size):
    audio_data, transcript_data = data
    padded_audio_data = pad_sequence(audio_data, max_length=max([len(seq) for seq in audio_data]))
    padded_transcript_data = pad_sequence(transcript_data, max_length=max([len(seq) for seq in transcript_data]))

    dataset = tf.data.Dataset.from_tensor_slices((padded_audio_data, padded_transcript_data))
    dataset = dataset.shuffle(buffer_size=len(audio_data))
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


if __name__ == "__main__":
    src_path = r'E:\Project\Python\MoChA-speech\data\raw_data\LibriSpeech'
    dst_path = r'E:\Project\Python\MoChA-speech\data\preprocessed_data'
    preprocess_librispeech(src_path, dst_path)
