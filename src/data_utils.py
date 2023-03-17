import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from audio_utils import process_audio_files
from txt_utils import process_transcripts, text_to_int_sequence


def preprocess_librispeech(src_path, dst_path):
    for dataset_type in ['train', 'dev', 'test']:
        process_audio_files(os.path.join(src_path, dataset_type),
                            os.path.join(dst_path, dataset_type),
                            dataset_type)
        process_transcripts(os.path.join(src_path, dataset_type),
                            os.path.join(dst_path, dataset_type),
                            dataset_type)


def data_generator(file_pattern):
    def parse_example(audio_file_path, transcript_file_path):
        audio_data = np.load(audio_file_path.numpy())
        with open(transcript_file_path.numpy(), 'r') as transcript_file:
            transcript_data = text_to_int_sequence(transcript_file.read().strip())
        return audio_data, transcript_data

    def tf_parse_example(audio_file_path, transcript_file_path):
        audio_data, transcript_data = tf.py_function(
            parse_example, [audio_file_path, transcript_file_path],
            [tf.float32, tf.int32])
        return audio_data, transcript_data

    file_pattern = os.path.join(file_pattern, '*/*.[nt][px][yt]')
    file_list = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    audio_files = file_list.filter(lambda x: tf.strings.regex_full_match(x, '.*\.npy'))
    transcript_files = file_list.filter(lambda x: tf.strings.regex_full_match(x, '.*\.txt'))

    dataset = tf.data.Dataset.zip((audio_files, transcript_files))
    dataset = dataset.map(tf_parse_example)
    return dataset


def count_examples(file_pattern):
    count = 0
    for file in tf.io.gfile.glob(file_pattern):
        if file.endswith(".npy"):
            count += 1
    return count

def load_data(data_path):
    def tqdm_generator(dataset, total_size):
        return tqdm(dataset, desc="Loading data", total=total_size, unit="example")

    train_data = data_generator(os.path.join(data_path, 'train'))
    dev_data = data_generator(os.path.join(data_path, 'dev'))
    test_data = data_generator(os.path.join(data_path, 'test'))

    train_size = count_examples(os.path.join(data_path, 'train/*/*.[nt][px][yt]'))
    dev_size = count_examples(os.path.join(data_path, 'dev/*/*.[nt][px][yt]'))
    test_size = count_examples(os.path.join(data_path, 'test/*/*.[nt][px][yt]'))

    return (tqdm_generator(train_data, train_size),
            tqdm_generator(dev_data, dev_size),
            tqdm_generator(test_data, test_size))


def pad_sequence(sequences, max_length=None, padding_value=0, pad_right=True):
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])

    padded_sequences = []
    for seq in sequences:
        padding = (0, max_length - len(seq)) if pad_right else (max_length - len(seq), 0)
        padded_seq = np.pad(seq, padding, mode='constant', constant_values=padding_value)
        padded_sequences.append(padded_seq)

    return padded_sequences


def create_dataset(data_generator, batch_size):
    dataset = data_generator.shuffle(buffer_size=1000)
    dataset = dataset.padded_batch(batch_size, padded_shapes=((None, None), (None,)), drop_remainder=True)
    return dataset


if __name__ == "__main__":
    src_path = r'E:\Project\Python\MoChA-speech\data\raw_data\LibriSpeech'
    dst_path = r'E:\Project\Python\MoChA-speech\data\preprocessed_data'
    preprocess_librispeech(src_path, dst_path)
