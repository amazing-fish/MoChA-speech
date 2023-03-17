import os
import sys
import numpy as np
import tensorflow as tf
from data_utils import load_data, create_dataset

from model import MoChAASR
from config import Config


def wer(predicted, ground_truth):
    import jiwer
    ground_truth = " ".join(ground_truth)
    predicted = " ".join(predicted)
    return jiwer.wer(ground_truth, predicted)


def evaluate(model, dataset):
    total_wer = 0
    total_samples = 0

    for batch in dataset:
        inputs, targets = batch
        predictions = model(inputs)
        decoded_predictions = decode_predictions(predictions)
        decoded_targets = decode_targets(targets)

        for pred, target in zip(decoded_predictions, decoded_targets):
            total_wer += wer(pred, target)
            total_samples += 1

    average_wer = total_wer / total_samples
    return average_wer


def decode_predictions(predictions):
    decoded_predictions = []
    for pred in predictions.numpy():
        decoded_pred = [Config.index2char[char_index] for char_index in np.argmax(pred, axis=-1)]
        decoded_predictions.append(decoded_pred)
    return decoded_predictions


def decode_targets(targets):
    decoded_targets = []
    for target in targets.numpy():
        decoded_target = [Config.index2char[char_index] for char_index in target]
        decoded_targets.append(decoded_target)
    return decoded_targets


if __name__ == "__main__":
    config = Config()
    _, _, test_data = load_data(config.data_path)
    test_dataset = create_dataset(test_data, config.batch_size)

    model = MoChAASR(config)
    model.load_weights(os.path.join(config.model_dir, "saved_models", "best_model.h5"))

    print("Evaluating the model...")
    test_wer = evaluate(model, test_dataset)
    print(f"Test WER: {test_wer:.4f}")
