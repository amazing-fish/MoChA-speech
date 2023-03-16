import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

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


if __name__ == "__main__":
    src_path = r'E:\Project\Python\MoChA-speech\data\raw_data\LibriSpeech'
    dst_path = r'E:\Project\Python\MoChA-speech\data\preprocessed_data'
    preprocess_librispeech(src_path, dst_path)
