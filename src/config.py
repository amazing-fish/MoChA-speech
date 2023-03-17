import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    @staticmethod
    def get_config():
        config = {
            # Data
            'data_path': os.path.join(PROJECT_ROOT, 'data', 'preprocessed_data'),
            'train_data': 'train',
            'dev_data': 'dev',
            'test_data': 'test',

            # Audio features
            'input_dim': 8,
            'output_dim': 29,

            # Model architecture
            'hidden_dim': 256,
            'num_layers': 2,
            'chunk_size': 8,
            'dropout_rate': 0.1,

            # Training
            'batch_size': 64,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'clip_gradient_norm': 5.0,

            # Evaluation
            'beam_width': 10,

            # Checkpoints and logging
            'model_dir': os.path.join(PROJECT_ROOT, 'models', 'saved_models'),
            'log_dir': os.path.join(PROJECT_ROOT, 'logs', 'training_logs')
        }

        return config
