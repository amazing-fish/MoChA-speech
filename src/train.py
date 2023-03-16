import os
import tensorflow as tf
from data_utils import load_data, create_dataset
from model import MoChAASR
from config import Config


def main():
    # Load data
    train_data, dev_data, test_data = load_data(Config.data_path)

    # Create datasets
    train_dataset = create_dataset(train_data, Config.batch_size)
    dev_dataset = create_dataset(dev_data, Config.batch_size)
    test_dataset = create_dataset(test_data, Config.batch_size)

    # Initialize model
    model = MoChAASR(Config.input_dim, Config.output_dim, Config.hidden_dim, Config.num_layers, Config.chunk_size,
                         Config.dropout_rate)

    # Set loss function and optimizer
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.learning_rate)

    # Checkpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, Config.model_dir, max_to_keep=3)

    # Training loop
    for epoch in range(Config.num_epochs):
        print(f'Epoch {epoch + 1}/{Config.num_epochs}')
        print('-' * 40)

        # Train
        model.train_on_batch(train_dataset, loss_fn, optimizer, Config.clip_gradient_norm)

        # Save model checkpoint
        checkpoint_manager.save()

        # Evaluate
        train_loss, train_wer = model.evaluate_on_batch(train_dataset, loss_fn)
        dev_loss, dev_wer = model.evaluate_on_batch(dev_dataset, loss_fn)
        test_loss, test_wer = model.evaluate_on_batch(test_dataset, loss_fn)

        print(f'Train loss: {train_loss:.4f}, WER: {train_wer:.2f}')
        print(f'Dev loss: {dev_loss:.4f}, WER: {dev_wer:.2f}')
        print(f'Test loss: {test_loss:.4f}, WER: {test_wer:.2f}')


if __name__ == '__main__':
    main()
