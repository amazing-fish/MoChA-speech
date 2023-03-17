import os
import string
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


CHAR_TO_INDEX = {char: index for index, char in
                 enumerate(" " + string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation)}

INDEX_TO_CHAR = {index: char for char, index in CHAR_TO_INDEX.items()}


def process_single_transcript_file(args):
    src_path, dst_path, file_path = args
    speaker_id = os.path.basename(os.path.dirname(file_path))
    dest_folder = os.path.join(dst_path, speaker_id)
    os.makedirs(dest_folder, exist_ok=True)
    with open(file_path, 'r') as transcript_file:
        lines = transcript_file.readlines()
        for line in lines:
            file_id, text = line.strip().split(' ', 1)
            dest_file_path = os.path.join(dest_folder, file_id + '.txt')
            with open(dest_file_path, 'w') as dest_file:
                dest_file.write(text)


def text_to_int_sequence(text):
    return [CHAR_TO_INDEX[char] for char in text if char in CHAR_TO_INDEX]


def process_transcripts(src_path, dst_path, dataset_type):
    transcript_files = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                transcript_files.append((src_path, dst_path, file_path))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_single_transcript_file, transcript_files), total=len(transcript_files), desc="Processing transcripts"))

