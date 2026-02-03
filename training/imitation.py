# training/imitation.py
"""
Imitation learning (behavior cloning) for warm-starting the policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import random


class ExpertTraceDataset(Dataset):
    """Dataset of expert traces for imitation learning."""
    
    def __init__(self,
                 traces: List[Dict],
                 config: Dict[str, Any]):
        """
        Initialize expert trace dataset.
        
        Args:
            traces: List of trace dictionaries containing observations and actions
            config: Configuration dictionary
        """
        self.traces = traces
        self.config = config
        
        # Flatten traces into individual (obs, action) pairs
        self.samples = []
        for trace in traces:
            observations = trace['observations']
            actions = trace['actions']
            for obs, action in zip(observations, actions):
                self.samples.append((obs, action))
                
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs, action = self.samples[idx]
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long)
        )


class ExpertTraceGenerator:
    """Generate expert traces by walking along ground truth centerlines."""
    
    # Direction mapping
    DIRECTIONS = np.array([
        [-1, 0],   # N
        [-1, 1],   # NE
        [0, 1],    # E
        [1, 1],    # SE
        [1, 0],    # S
        [1, -1],   # SW
        [0, -1],   # W
        [-1, -1],  # NW
    ])
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.obs_size = config.get('environment', {}).get('observation_size', 65)
        
        from environment.observation import ObservationBuilder
        self.obs_builder = ObservationBuilder(config)
        
    def generate_traces(self,
                        image: np.ndarray,
                        centerline: np.ndarray,
                        vesselness: Optional[np.ndarray] = None,
                        max_traces: int = 100) -> List[Dict]:
        """
        Generate expert traces from image and centerline.
        
        Args:
            image: RGB image (H, W, 3)
            centerline: Binary centerline (H, W)
            vesselness: Optional vesselness response
            max_traces: Maximum number of traces to generate
            
        Returns:
            List of trace dictionaries
        """
        from data.centerline_extraction import CenterlineExtractor
        
        extractor = CenterlineExtractor()
        
        # Convert centerline to graph
        graph = extractor.skeleton_to_graph(centerline)
        
        if len(graph.nodes) == 0:
            return []
        
        # Generate traces by walking edges
        traces = []
        visited_edges = set()
        
        # Start from endpoints
        endpoints = [n for n in graph.nodes if graph.degree(n) == 1]
        if not endpoints:
            endpoints = [list(graph.nodes)[0]]
        
        for start_node in endpoints:
            # DFS from this endpoint
            stack = [(start_node, None, [])]
            
            while stack and len(traces) < max_traces:
                current, prev_edge, path = stack.pop()
                
                for neighbor in graph.neighbors(current):
                    edge_key = tuple(sorted([current, neighbor]))
                    
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        
                        edge_data = graph.get_edge_data(current, neighbor)
                        edge_path = edge_data.get('path', [])
                        
                        if edge_path:
                            # Generate trace for this edge
                            trace = self._trace_path(
                                image, centerline, vesselness, edge_path
                            )
                            if trace:
                                traces.append(trace)
                        
                        stack.append((neighbor, edge_key, edge_path))
        
        return traces
    
    def _trace_path(self,
                    image: np.ndarray,
                    centerline: np.ndarray,
                    vesselness: Optional[np.ndarray],
                    path: List[Tuple[int, int]]) -> Optional[Dict]:
        """
        Generate observation-action pairs for a path.
        
        Args:
            image: RGB image
            centerline: Centerline mask
            vesselness: Optional vesselness
            path: List of (y, x) positions
            
        Returns:
            Dictionary with observations and actions
        """
        if len(path) < 2:
            return None
        
        h, w = image.shape[:2]
        half = self.obs_size // 2
        
        observations = []
        actions = []
        visited_mask = np.zeros((h, w), dtype=np.float32)
        prev_direction = None
        
        for i in range(len(path) - 1):
            current_pos = np.array(path[i])
            next_pos = np.array(path[i + 1])
            
            # Check if position is valid
            y, x = current_pos
            if y < half or y >= h - half or x < half or x >= w - half:
                continue
            
            # Mark visited
            visited_mask[y, x] = 1.0
            
            # Compute action (direction to next position)
            direction = next_pos - current_pos
            action = self._direction_to_action(direction)
            
            if action is None:
                continue
            
            # Build observation
            obs = self.obs_builder.build(
                image=image,
                visited_mask=visited_mask,
                vesselness=vesselness,
                position=current_pos,
                prev_direction=prev_direction
            )
            
            observations.append(obs)
            actions.append(action)
            prev_direction = action
        
        if len(observations) == 0:
            return None
        
        return {
            'observations': observations,
            'actions': actions,
            'path': path
        }
    
    def _direction_to_action(self, direction: np.ndarray) -> Optional[int]:
        """Convert direction vector to action index."""
        # Normalize direction
        if np.linalg.norm(direction) == 0:
            return None
        
        direction = direction / np.linalg.norm(direction)
        
        # Find closest matching direction
        best_action = None
        best_similarity = -1
        
        for i, d in enumerate(self.DIRECTIONS):
            d_norm = d / np.linalg.norm(d)
            similarity = np.dot(direction, d_norm)
            if similarity > best_similarity:
                best_similarity = similarity
                best_action = i
        
        return best_action


class ImitationLearner:
    """Behavior cloning trainer."""
    
    def __init__(self,
                 policy: nn.Module,
                 config: Dict[str, Any],
                 device: torch.device):
        self.policy = policy
        self.config = config
        self.device = device
        
        il_config = config.get('training', {}).get('imitation', {})
        
        self.epochs = il_config.get('epochs', 50)
        self.batch_size = il_config.get('batch_size', 256)
        self.lr = il_config.get('learning_rate', 1e-3)
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
        
    def train(self,
              dataset: ExpertTraceDataset,
              val_dataset: Optional[ExpertTraceDataset] = None) -> Dict[str, List]:
        """
        Train policy using behavior cloning.
        
        Args:
            dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            Training history
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.epochs):
            # Training
            self.policy.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.epochs}')
            
            for obs, actions in pbar:
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                action_logits, _, _ = self.policy.forward(obs)
                
                # Compute loss
                loss = F.cross_entropy(action_logits, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                
                # Metrics
                train_losses.append(loss.item())
                pred = action_logits.argmax(dim=1)
                train_correct += (pred == actions).sum().item()
                train_total += actions.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.2%}'
                })
            
            self.scheduler.step()
            
            avg_train_loss = np.mean(train_losses)
            train_acc = train_correct / train_total
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if val_dataset is not None:
                val_loss, val_acc = self.evaluate(val_dataset)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f'Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, '
                      f'train_acc={train_acc:.2%}, val_loss={val_loss:.4f}, '
                      f'val_acc={val_acc:.2%}')
            else:
                print(f'Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, '
                      f'train_acc={train_acc:.2%}')
        
        return history
    
    def evaluate(self, dataset: ExpertTraceDataset) -> Tuple[float, float]:
        """Evaluate on dataset."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        self.policy.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for obs, actions in dataloader:
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                
                action_logits, _, _ = self.policy.forward(obs)
                
                loss = F.cross_entropy(action_logits, actions)
                total_loss += loss.item() * actions.size(0)
                
                pred = action_logits.argmax(dim=1)
                correct += (pred == actions).sum().item()
                total += actions.size(0)
        
        return total_loss / total, correct / total
