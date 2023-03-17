import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def process_single_audio_file(args):
    src_path, dst_path, file_path = args
    mfcc = extract_mfcc(file_path)
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    speaker_id = os.path.basename(os.path.dirname(file_path))
    dest_folder = os.path.join(dst_path, speaker_id)
    os.makedirs(dest_folder, exist_ok=True)
    np.save(os.path.join(dest_folder, file_id + '.npy'), mfcc)


def extract_mfcc(file_path, n_mfcc=8):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T

    # Standardize MFCC features
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc)

    return mfcc


def process_audio_files(src_path, dst_path, dataset_type):
    audio_files = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                audio_files.append((src_path, dst_path, file_path))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_audio_file, audio_files), total=len(audio_files), desc="Processing "
                                                                                                     "audio files"))
