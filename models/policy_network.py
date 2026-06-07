"""Actor-Critic policy network for the vessel-tracing RL agent.

CNN/ResNet encoder → optional LSTMCell → actor + critic heads. ``forward`` runs one timestep
(rollout); ``forward_sequence`` runs a T-step chunk with done masks (PPO training).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_in_channels(config: Dict[str, Any]) -> int:
    """Return the observation channel count; thin shim over ``ObservationBuilder.n_channels``."""
    from environment.observation import ObservationBuilder

    return ObservationBuilder.n_channels(config)


def _junction_channel_idx(config: Dict[str, Any]) -> Optional[int]:
    """Return the observation channel index of the junction map, or None if disabled.

    Used by the PPO trainer to read junction supervision targets from the stored obs batch.
    """
    from environment.observation import ObservationBuilder

    return ObservationBuilder.junction_channel_idx(config)


class CNNEncoder(nn.Module):
    """5-layer dilated CNN encoder with attention+max pooling whose RF spans the full crop.

    GroupNorm (not BatchNorm) keeps rollout-time eval() and update-time train() features
    identical, so PPO importance ratios stay clean; GAP replaces a large Linear bottleneck.
    """

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0):
        """Build the dilated conv stack, attention/max pooling, and the projection FC."""
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),  # 65->33, RF=5
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 33->17, RF=9
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 17->17, RF=17
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),  # 17->9, RF=33
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),  # 9->9, RF=81
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )

        self.attn_conv = nn.Conv2d(128, 1, kernel_size=1, bias=True)
        self.gap_max = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),  # 128 attended + 128 max
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv stack, fuse spatial-attention mean with global max, project; returns (B, hidden_dim)."""
        x = self.conv_layers(x)
        B = x.shape[0]
        # Soft spatial attention over the actual feature-map size (any observation_size).
        attn = self.attn_conv(x)
        attn = F.softmax(attn.view(B, -1), dim=-1).view(B, 1, *x.shape[2:])
        x_attn = (x * attn).sum(dim=(2, 3))  # attention-weighted mean
        x_max = self.gap_max(x).flatten(1)
        x = torch.cat([x_attn, x_max], dim=1)
        x = self.dropout(x)
        return self.fc(x)


class ResNetEncoder(nn.Module):
    """Lightweight ResNet-style encoder (stem + two residual blocks) with attention+max pooling."""

    class ResBlock(nn.Module):
        """Two-conv residual block with GroupNorm and a skip connection."""

        def __init__(self, channels: int):
            """Build the two GroupNorm conv layers of the block."""
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ReLU(x + block(x))."""
            return F.relu(x + self.block(x))

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0):
        """Build the stem, residual blocks, attention/max pooling, and projection FC."""
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 32, 5, stride=2, padding=2, bias=False), nn.GroupNorm(8, 32), nn.ReLU())
        self.layer1 = self.ResBlock(32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.layer2 = self.ResBlock(64)
        self.attn_conv = nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.gap_max = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())  # 64 attended + 64 max
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run stem and residual blocks, fuse attention mean with global max, project; returns (B, hidden_dim)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        B = x.shape[0]
        attn = self.attn_conv(x)
        attn = F.softmax(attn.view(B, -1), dim=-1).view(B, 1, *x.shape[2:])
        x_attn = (x * attn).sum(dim=(2, 3))
        x_max = self.gap_max(x).flatten(1)
        x = torch.cat([x_attn, x_max], dim=1)
        x = self.dropout(x)
        return self.fc(x)


class LSTMHead(nn.Module):
    """Single LSTMCell (with input layer-norm) and a learnable initial hidden state.

    Uses LSTMCell because RL rollout is step-by-step; ``forward_sequence`` loops it for
    training. The learnable ``(init_h, init_c)`` is the reset target on episode start / done.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """Build the LayerNorm, LSTMCell, and learnable initial-state parameters."""
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.init_h = nn.Parameter(torch.zeros(hidden_dim))
        self.init_c = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Advance the LSTMCell one step, using the learnable init state when ``state`` is None.

        Returns ``(h, (h, c))``.
        """
        x = self.ln(x)
        if state is None:
            B = x.size(0)
            h = self.init_h.unsqueeze(0).expand(B, -1).contiguous()
            c = self.init_c.unsqueeze(0).expand(B, -1).contiguous()
        else:
            h, c = state
        h, c = self.lstm(x, (h, c))
        return h, (h, c)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic policy: encoder (CNN|ResNet) → optional LSTMCell → actor + critic heads."""

    N_ACTIONS = 9  # N, NE, E, SE, S, SW, W, NW + STOP (index 8)

    def __init__(self, config: Dict[str, Any]):
        """Build encoder, optional LSTM, actor/critic/junction heads from ``config['policy']``."""
        super().__init__()
        policy_cfg = config.get('policy', {})

        hidden_dim = policy_cfg.get('hidden_dim', 128)
        lstm_hidden = policy_cfg.get('lstm_hidden', 128)
        head_hidden = policy_cfg.get('head_hidden', 128)
        self.use_lstm = policy_cfg.get('use_lstm', False)
        dropout = policy_cfg.get('dropout', 0.0)
        encoder_type = policy_cfg.get('encoder_type', 'cnn')

        # Spatial channels go through the CNN; scalar (broadcast-constant) channels
        # (prev_action, topology_memory) bypass it and feed the MLP heads directly.
        from environment.observation import ObservationBuilder

        in_channels_spatial = ObservationBuilder.n_spatial_channels(config)
        self.n_scalar_channels = ObservationBuilder.n_scalar_channels(config)

        if encoder_type == 'resnet':
            self.encoder = ResNetEncoder(in_channels_spatial, hidden_dim, dropout)
        else:
            self.encoder = CNNEncoder(in_channels_spatial, hidden_dim, dropout)

        # LSTM operates on encoder output only; scalars are concatenated after it so the
        # heads see current scalars directly while LSTM width stays at hidden_dim.
        if self.use_lstm:
            self.lstm_head = LSTMHead(hidden_dim, lstm_hidden)
            feature_dim_post_lstm = lstm_hidden
        else:
            self.lstm_head = None
            feature_dim_post_lstm = hidden_dim
        head_input_dim = feature_dim_post_lstm + self.n_scalar_channels

        # LayerNorm + Tanh in the heads stabilises PPO against intra-epoch feature drift.
        self.actor_head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden), nn.LayerNorm(head_hidden), nn.Tanh(), nn.Linear(head_hidden, self.N_ACTIONS)
        )
        self.value_head = nn.Sequential(nn.Linear(head_input_dim, head_hidden), nn.LayerNorm(head_hidden), nn.Tanh(), nn.Linear(head_hidden, 1))

        env_cfg = config.get('environment', {})
        use_junction_aux = policy_cfg.get('use_junction_aux', True) and env_cfg.get('use_junction', True)
        if use_junction_aux:
            self.junction_head: Optional[nn.Sequential] = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        else:
            self.junction_head = None

        self._init_weights()

    def _init_weights(self):
        """Orthogonal-init conv/linear layers, with a near-zero actor, negative STOP bias, and zeroed attention.

        The near-zero actor gives a near-uniform initial policy, the negative STOP-logit bias
        avoids premature "always STOP" collapse, and zeroed attention starts as average pooling.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        if self.N_ACTIONS == 9:
            with torch.no_grad():
                self.actor_head[-1].bias[8] = -1.0

        # Zero attention weights → uniform softmax (≡ average pooling) at init.
        if hasattr(self.encoder, 'attn_conv'):
            nn.init.zeros_(self.encoder.attn_conv.weight)
            nn.init.zeros_(self.encoder.attn_conv.bias)

    def init_hidden(self, batch_size: int = 1, device: Union[torch.device, str] = 'cpu'):
        """Return the learnable initial LSTM state broadcast over the batch, or None if LSTM is off.

        Detached, so rollout inference doesn't pin a gradient graph; ``forward_sequence``
        references the same parameters directly so training gradients still flow.
        """
        if not self.use_lstm:
            return None
        h = self.lstm_head.init_h.detach().to(device).unsqueeze(0).expand(batch_size, -1).contiguous()
        c = self.lstm_head.init_c.detach().to(device).unsqueeze(0).expand(batch_size, -1).contiguous()
        return h, c

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Split ``obs`` into (spatial, scalar) along channels; scalars read from the (0, 0) pixel.

        Returns ``scalars=None`` when no scalar channels are configured.
        """
        if self.n_scalar_channels == 0:
            return obs, None
        n = self.n_scalar_channels
        return obs[:, :-n], obs[:, -n:, 0, 0]

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Return CNN encoder features (pre-LSTM) for auxiliary losses, stripping the scalar tail."""
        spatial, _ = self._split_obs(obs)
        return self.encoder(spatial)

    def forward(
        self, obs: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Process one timestep.

        Args:
            obs: (B, C, H, W) — C = spatial + scalar channels.
            state: (h, c) each (B, lstm_hidden) or None.

        Returns:
            ``(logits (B, N_ACTIONS), values (B,), updated state or None)``.
        """
        spatial, scalars = self._split_obs(obs)
        features = self.encoder(spatial)

        if self.lstm_head is not None:
            features, state = self.lstm_head(features, state)

        if scalars is not None:
            features = torch.cat([features, scalars], dim=-1)

        logits = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values, state

    def forward_sequence(
        self, obs_seq: torch.Tensor, init_state: Optional[Tuple[torch.Tensor, torch.Tensor]], dones: torch.Tensor, return_enc_features: bool = False
    ):
        """Process a T-step chunk for PPO, resetting LSTM state after each done.

        Args:
            obs_seq: (T, B, C, H, W) sequential observations.
            init_state: (h, c) each (B, lstm_hidden).
            dones: (T, B) float — 1.0 where an episode ended (next step starts fresh).
            return_enc_features: also return encoder features (T*B, hidden_dim) for aux losses.

        Returns:
            ``(all_logits (T, B, N_ACTIONS), all_values (T, B))``, plus encoder features when
            requested. Without LSTM, all T*B observations are batch-forwarded in one pass.
        """
        T, B = obs_seq.shape[:2]

        # Slice scalars off once: spatial goes through the CNN, scalars rejoin per step.
        flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
        flat_spatial, flat_scalars = self._split_obs(flat)
        if flat_scalars is not None:
            scalars_seq = flat_scalars.view(T, B, -1)
        else:
            scalars_seq = None

        if not self.use_lstm:
            features = self.encoder(flat_spatial)
            if flat_scalars is not None:
                features = torch.cat([features, flat_scalars], dim=-1)
            logits = self.actor_head(features).view(T, B, -1)
            values = self.value_head(features).squeeze(-1).view(T, B)
            if return_enc_features:
                # Aux heads see pre-concat encoder features only (what the CNN sees).
                enc_only = features[:, : -self.n_scalar_channels] if self.n_scalar_channels else features
                return logits, values, enc_only
            return logits, values

        # Recurrent path: encode all frames in one batch, then loop the LSTM.
        all_features = self.encoder(flat_spatial).view(T, B, -1)

        h, c = init_state
        all_logits = []
        all_values = []

        for t in range(T):
            # Reset to the learnable init state where the previous step ended, so gradients
            # reach init_h/init_c through every done event in the chunk.
            if t > 0:
                done_mask = dones[t - 1].unsqueeze(-1)  # (B, 1)
                init_h_b = self.lstm_head.init_h.unsqueeze(0).expand_as(h)
                init_c_b = self.lstm_head.init_c.unsqueeze(0).expand_as(c)
                h = h * (1.0 - done_mask) + init_h_b * done_mask
                c = c * (1.0 - done_mask) + init_c_b * done_mask

            features_t, (h, c) = self.lstm_head(all_features[t], (h, c))
            if scalars_seq is not None:
                features_t = torch.cat([features_t, scalars_seq[t]], dim=-1)
            all_logits.append(self.actor_head(features_t))
            all_values.append(self.value_head(features_t).squeeze(-1))

        if return_enc_features:
            return (torch.stack(all_logits), torch.stack(all_values), all_features.reshape(T * B, -1))
        return torch.stack(all_logits), torch.stack(all_values)

    def get_action_and_value(
        self, obs: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Sample an action and return ``(action, log_prob, entropy, value, state)`` for rollout."""
        logits, values, state = self.forward(obs, state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return (action, dist.log_prob(action), dist.entropy(), values, state)

    def get_value(self, obs: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Return the critic value only (for GAE bootstrap)."""
        _, values, _ = self.forward(obs, state)
        return values
