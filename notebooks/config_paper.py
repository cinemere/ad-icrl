"""Dark Room hyperparameters from paper.
"""
from dataclasses import dataclass
from typing import Literal

@dataclass
class ArchParams:
    embedding_dim: int = 64
    """Embedding dim"""
    n_layers: int = 4
    """Number of layers"""
    n_heads: int = 4
    """Number of heads"""
    dim_ff: int = 2048
    """Feedforward dimd"""
    pos_enc: Literal["absolute"] = "absolute"
    """Position Encodings""" 
    ln_placem: Literal["postnorm"] = "postnorm"
    """Layer Norm Placement"""
    dropout: float = 0.1
    """Dropout Rate"""
    dropout_attn: float = 0.5
    """Attention Dropout Rate""" 
    mask_prob: float = 0.3
    """Sequence Mask Prob""" 
    smoothing: float = 0
    """Label Smoothing alpha""" 
    
@dataclass
class OptParams:
    batch_size: int = 128
    """Batch Size"""
    optim: Literal["adam"] = "adam"
    """Optimizer"""
    beta_1: float = 0.9
    """β1"""
    beta_2: float = 0.99
    """β2"""
    clip_norm: float = 1
    """Gradient Clip Norm Threshold"""
    lr_sheduler: Literal["no", "cos"] = "cos"
    """Learning Rate Schedule"""
    lr_init: float = 2e-6
    """Initial Value"""
    lr_peak: float = 3e-4
    """Peak Value"""
    
    
@dataclass
class A3CParams:
    batch_size: int = 100
    """Batch Size (Num. Actors)"""
    lambda_val: float = 0.95
    """λ"""
    agent_discount: float = 0.99
    """Agent Discount"""
    entropy_bonus_weight: float = 0.01
    """Entropy Bonus Weight"""
    mlp_layers: int = 3
    """MLP Layers"""
    mlp_hidden_dim: int = 128
    """MLP Hidden Dim"""    
    optim: Literal["adam"] = "adam"
    """Optimizer"""
    beta_1: float = 0.9
    """β1"""
    beta_2: float = 0.999
    """β2"""
    epsilon: float = 1e-6
    """Epsilon"""
    lr: float = 1e-4
    """Learning Rate"""
    
@dataclass
class RL2Params:
    rl_algorithm: Literal["a3c"] = "a3c"
    """RL Algorithm"""
    learning_rate: float = 3e-4
    """Learning Rate"""
    batch_size: int = 256
    """Batch Size"""
    unroll_length: int = 20
    """Unroll Length"""
    lstm_hidden_dim: int = 256
    """LSTM Hidden Dim."""
    lstm_num_layers: int = 2
    """LSTM Number of Layers"""
    episodes_per_trial: int = 10
    """Episodes Per Trial"""
    