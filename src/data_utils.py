import os
import librosa
import numpy as np
from tqdm import tqdm

def preprocess_data(data_type, source_dir, target_dir, n_mels=80, sr=16000, n_fft=400, hop_length=160, top_db=20):
    """
    Preprocess the LibriSpeech dataset and store the log-mel features as NumPy arrays.

    Args:
        data_type (str): "train", "dev", or "test".
        source_dir (str): Path to the raw data directory.
        target_dir (str): Path to the preprocessed data directory.
        n_mels (int): Number of mel filters.
        sr (int): Sampling rate of audio.
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length for FFT.
        top_db (int): Top decibels to remove noise from audio.
    """

    data_path = os.path.join(source_dir, data_type)
    output_path = os.path.join(target_dir, data_type)
    os.makedirs(output_path, exist_ok=True)

    for speaker in tqdm(os.listdir(data_path)):
        speaker_path = os.path.join(data_path, speaker)

        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)
            chapter_output_path = os.path.join(output_path, f"{speaker}-{chapter}")
            os.makedirs(chapter_output_path, exist_ok=True)

            for file in os.listdir(chapter_path):
                if file.endswith(".txt"):
                    continue

                audio_path = os.path.join(chapter_path, file)
                audio, _ = librosa.load(audio_path, sr=sr)
                mel_spec = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                log_mel_spec = librosa.power_to_db(mel_spec, top_db=top_db)
                np.save(os.path.join(chapter_output_path, f"{os.path.splitext(file)[0]}.npy"), log_mel_spec)

def process_transcripts(data_type, source_dir, target_dir):
    """
    Process the LibriSpeech dataset transcripts and store them as text files.

    Args:
        data_type (str): "train", "dev", or "test".
        source_dir (str): Path to the raw data directory.
        target_dir (str): Path to the preprocessed data directory.
    """

    data_path = os.path.join(source_dir, data_type)
    output_path = os.path.join(target_dir, data_type)
    os.makedirs(output_path, exist_ok=True)

    for speaker in tqdm(os.listdir(data_path)):
        speaker_path = os.path.join(data_path, speaker)

        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)
            chapter_output_path = os.path.join(output_path, f"{speaker}-{chapter}")
            os.makedirs(chapter_output_path, exist_ok=True)

            for file in os.listdir(chapter_path):
                if not file.endswith(".txt"):
                    continue

                transcript_path = os.path.join(chapter_path, file)

                with open(transcript_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    file_id, transcript = line.strip().split(" ", 1)
                    transcript_output_path = os.path.join(chapter_output_path, f"{file_id}.txt")

                    with open(transcript_output_path, "w") as f_out:
                        f_out.write(transcript)

if __name__ == "__main__":
    source_dir = "data/raw_data"
    target_dir = "data/preprocessed_data"

    for data_type in ["train", "dev", "test"]:
        print(f"Preprocessing {data_type} data...")
        preprocess_data(data_type, source_dir, target_dir)
        process_transcripts(data_type, source_dir, target_dir)
        print(f"{data_type.capitalize()} data preprocessing completed!")

