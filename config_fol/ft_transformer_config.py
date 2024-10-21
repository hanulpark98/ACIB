from .base_config import base_config
from copy import deepcopy

ft_transformer_config = deepcopy(base_config)

ft_transformer_config.model.search_space = {
    
    'num_layers' : ['suggest_int', ['num_layers', 2, 6]],  # Number of transformer layers
    'num_heads' : ['suggest_int', ['num_heads', 2, 8]],  # Number of attention heads
    'embed_dim' : ['suggest_int', ['embed_dim', 64, 512]],  # Embedding dimension for the transformer
    'ffn_dim' : ['suggest_int', ['ffn_dim', 128, 1024]],  # Feed-forward network dimension
    'dropout' : ['suggest_float', ['dropout', 0.1, 0.5]],  # Dropout rate
    'attention_dropout' : ['suggest_float', ['attention_dropout', 0.1, 0.5]],  # Dropout rate for attention layers

    'learning_rate' : ['suggest_float', ['learning_rate', 1e-5, 1e-2, 'log']],  # Learning rate (log scale)
    'weight_decay' : ['suggest_float', ['weight_decay', 1e-6, 1e-2, 'log']],  # Weight decay for regularization

    'batch_size' : ['suggest_categorical', ['batch_size', [32, 64, 128, 256]]],  # Batch size for training
    'optimizer' : ['suggest_categorical', ['optimizer', ['adam', 'adamw', 'sgd']]],  # Optimizer choice
    'activation_function' : ['suggest_categorical', ['activation_function', ['relu', 'gelu', 'leaky_relu']]],  # Activation function for transformer layers

    'warmup_steps' : ['suggest_int', ['warmup_steps', 100, 1000]],  # Warmup steps for learning rate scheduler
    'scheduler_type' : ['suggest_categorical', ['scheduler_type', ['linear', 'cosine']]]  # Learning rate scheduler type
}