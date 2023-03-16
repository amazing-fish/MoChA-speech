class Config:
    # Data
    data_path = 'data/preprocessed_data/'
    train_data = 'train'
    dev_data = 'dev'
    test_data = 'test'

    # Audio features
    input_dim = 13  # Number of MFCC features
    output_dim = 29  # Number of output classes (characters, including a special 'blank' character)

    # Model architecture
    hidden_dim = 256  # Hidden units in RNN layers
    num_layers = 2  # Number of RNN layers in both encoder and decoder
    chunk_size = 8  # MoChA attention chunk size
    dropout_rate = 0.1  # Dropout rate for RNN layers

    # Training
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    clip_gradient_norm = 5.0  # Gradient clipping to avoid exploding gradients

    # Evaluation
    beam_width = 10  # Beam width for beam search during evaluation

    # Checkpoints and logging
    model_dir = 'models/saved_models'
    log_dir = 'logs/training_logs'
