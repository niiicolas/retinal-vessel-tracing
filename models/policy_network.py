# models/policy_network.py
"""
Policy and value networks for PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np


class CNNEncoder(nn.Module):
    """CNN encoder for processing local observations."""
    
    def __init__(self, 
                 in_channels: int,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        features = self.conv_layers(x)
        features = features.view(batch_size, -1)
        return self.fc(features)


class ResNetEncoder(nn.Module):
    """ResNet-based encoder using pretrained weights."""
    
    def __init__(self, 
                 in_channels: int,
                 hidden_dim: int = 256,
                 pretrained: bool = True):
        super().__init__()
        
        import torchvision.models as models
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for different number of input channels
        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize new channels with mean of old RGB weights
            with torch.no_grad():
                if in_channels > 3:
                    resnet.conv1.weight[:, :3] = old_conv.weight
                    resnet.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True).expand(-1, in_channels - 3, -1, -1)
                else:
                    resnet.conv1.weight = old_conv.weight[:, :in_channels]
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        features = self.features(x)
        features = features.view(batch_size, -1)
        return self.fc(features)


class LSTMHead(nn.Module):
    """LSTM head for handling partial observability."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            hidden: Optional hidden state tuple (h, c)
            
        Returns:
            output: LSTM output
            hidden: New hidden state
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        output, hidden = self.lstm(x, hidden)
        
        # Return last timestep output
        return output[:, -1], hidden
    
    def get_initial_hidden(self, batch_size: int, device: torch.device):
        """Get initial hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Architecture:
    - CNN encoder for processing local observations
    - Optional LSTM head for temporal information
    - Separate policy (actor) and value (critic) heads
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        policy_config = config.get('policy', {})
        env_config = config.get('environment', {})
        
        self.hidden_dim = policy_config.get('hidden_dim', 256)
        self.lstm_hidden = policy_config.get('lstm_hidden', 128)
        self.use_lstm = policy_config.get('use_lstm', True)
        self.dropout = policy_config.get('dropout', 0.1)
        encoder_type = policy_config.get('encoder_type', 'cnn')
        
        # Compute number of input channels
        # RGB (3) + visited (1) + direction (8) + optional vesselness (1)
        in_channels = 3 + 1 + 8
        if env_config.get('use_vesselness', True):
            in_channels += 1
        
        # Build encoder
        if encoder_type == 'resnet':
            self.encoder = ResNetEncoder(
                in_channels=in_channels,
                hidden_dim=self.hidden_dim,
                pretrained=policy_config.get('pretrained', True)
            )
        else:
            self.encoder = CNNEncoder(
                in_channels=in_channels,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            )
        
        # Build LSTM head if needed
        if self.use_lstm:
            self.lstm = LSTMHead(
                input_dim=self.encoder.output_dim,
                hidden_dim=self.lstm_hidden
            )
            head_input_dim = self.lstm.output_dim
        else:
            self.lstm = None
            head_input_dim = self.encoder.output_dim
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # 8 directions + STOP
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights."""
        for module in [self.policy_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, 
                obs: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor (batch, channels, height, width)
            hidden: Optional LSTM hidden state
            
        Returns:
            action_logits: Logits for action distribution
            value: Estimated state value
            new_hidden: New LSTM hidden state (if using LSTM)
        """
        # Encode observation
        features = self.encoder(obs)
        
        # Process through LSTM if enabled
        new_hidden = None
        if self.use_lstm and self.lstm is not None:
            features, new_hidden = self.lstm(features, hidden)
        
        # Get policy and value
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return action_logits, value.squeeze(-1), new_hidden
    
    def get_action_and_value(self, 
                              obs: torch.Tensor,
                              hidden: Optional[Tuple] = None,
                              action: Optional[torch.Tensor] = None
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation tensor
            hidden: Optional LSTM hidden state
            action: Optional action to evaluate (for training)
            
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: Estimated value
            new_hidden: New LSTM hidden state
        """
        action_logits, value, new_hidden = self.forward(obs, hidden)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value, new_hidden
    
    def get_value(self, 
                  obs: torch.Tensor,
                  hidden: Optional[Tuple] = None) -> torch.Tensor:
        """Get only the value estimate."""
        _, value, _ = self.forward(obs, hidden)
        return value
