import os
import tensorflow as tf
from tqdm import tqdm

from data_utils import load_data, create_dataset
from model import MoChAASR
from config import Config

# Load configuration
config = Config.get_config()


def main():
    # Load data
    print("Loading data...")
    train_data, dev_data = load_data(config['data_path'])[:-1]
    print("Data loaded successfully.")

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dataset(train_data, config['batch_size'])
    dev_dataset = create_dataset(dev_data, config['batch_size'])
    print("Datasets created successfully.")

    # Initialize model
    print("Initializing MoChA ASR model...")
    model = MoChAASR(config['input_dim'], config['output_dim'], config['hidden_dim'], config['num_layers'],
                     config['chunk_size'],
                     config['dropout_rate'])
    print("MoChA ASR model initialized successfully.")

    # Set loss function and optimizer
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    # Checkpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config['model_dir'], max_to_keep=3)

    # Training loop
    for epoch in range(config['num_epochs']):
        print(f'Epoch {epoch + 1}/{config["num_epochs"]}')
        print('-' * 40)

        # Train
        print("Training on batch...")
        for batch in tqdm(train_dataset, desc="Training", unit="batch"):
            model.train_on_batch(batch, loss_fn, optimizer, config['clip_gradient_norm'])
        print("Training completed.")

        # Save model checkpoint
        print("Saving model checkpoint...")
        checkpoint_manager.save()
        print("Model checkpoint saved successfully.")

        # Evaluate
        print("Evaluating on batches...")
        train_loss, train_wer = model.evaluate_on_batch(tqdm(train_dataset, desc="Train Eval", unit="batch"), loss_fn)
        dev_loss, dev_wer = model.evaluate_on_batch(tqdm(dev_dataset, desc="Dev Eval", unit="batch"), loss_fn)
        print("Evaluation completed.")

        print(f'Train loss: {train_loss:.4f}, WER: {train_wer:.2f}')
        print(f'Dev loss: {dev_loss:.4f}, WER: {dev_wer:.2f}')

    # Load test data after training
    print("Loading test data...")
    test_data = load_data(config['data_path'])[-1]  # 仅加载测试集
    print("Test data loaded successfully.")
    test_dataset = create_dataset(test_data, config['batch_size'])

    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_wer = model.evaluate_on_batch(tqdm(test_dataset, desc="Test Eval", unit="batch"), loss_fn)
    print("Test set evaluation completed.")
    print(f'Test loss: {test_loss:.4f}, WER: {test_wer:.2f}')


if __name__ == '__main__':
    main()
