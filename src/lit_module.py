"""
Lightning module for Rubik's Cube RL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import omegaconf
import lightning as L
from typing import override
from src import rubiks_dataset


class LitModule(L.LightningModule):
    """
    Lightning module for training Rubik's Cube RL models.
    """
    
    def __init__(self,config: omegaconf.DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = omegaconf.OmegaConf.create(config)
        
        self.model = self.create_model(self.config.model)

    def create_model(self, config: omegaconf.DictConfig) -> nn.Module:
        """
        Create a model with residual blocks.
        """
        layers = []
        
        # Input projection layer
        layers.append(nn.Linear(config.input_size, config.hidden_size))
        layers.append(nn.ReLU())
        
        # Add residual blocks
        for _ in range(config.num_blocks):
            layers.append(ResidualBlock(config.hidden_size))
        
        # Output projection layer
        layers.append(nn.Linear(config.hidden_size, config.output_size))
        
        return nn.Sequential(*layers)

    
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    @override
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:

        next_states = batch["next_states"].float()
        current_state = batch["current_state"].float()
        max_steps_taken = batch["max_steps_taken"]

        # Get the minimum steps to go from the next states.
        with torch.no_grad():
            steps_to_go = self.get_minimimun_steps_to_go_from_next_states(next_states)
            steps_to_go_targets = torch.minimum(steps_to_go + 1, max_steps_taken)

        steps_to_go_logits = self.model(current_state)
        loss = F.cross_entropy(steps_to_go_logits, steps_to_go_targets)


        steps_to_go_pred = torch.argmax(steps_to_go_logits, dim=1)
        avg_steps_to_go_pred = steps_to_go_pred.float().mean()
        step_to_go_error = abs(steps_to_go_pred - steps_to_go_targets).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_step_to_go_error", step_to_go_error, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_avg_steps_to_go_pred", avg_steps_to_go_pred, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
    def get_minimimun_steps_to_go_from_next_states(self, next_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_next_states, num_features = next_states.shape

        next_states = next_states.reshape(batch_size * num_next_states, num_features)

        steps_to_go_logits = self.model(next_states)

        steps_to_go_probs = F.softmax(steps_to_go_logits, dim=1)

        # For each state, sample steps_to_go from the softmax distribution.
        # Ie the first class is 0 steps to go, the second class is 1 step to go, etc.
        steps_to_go = torch.multinomial(steps_to_go_probs, num_samples=1).squeeze(-1)

        steps_to_go = steps_to_go.reshape(batch_size, num_next_states)

        # Get the steps to go from the best next state.
        steps_to_go = steps_to_go.min(dim=1, keepdim=False).values

        return steps_to_go


    
    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adam(self.parameters(), lr=self.config.train.learning_rate)
        return optimizer

    @override
    def train_dataloader(self) -> torch.utils.data.DataLoader:

        dataset = rubiks_dataset.RubiksDataset(self.config.data)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            persistent_workers=True,
        )
        return dataloader

class ResidualBlock(nn.Module):
    """
    Simple residual block with two linear layers, normalization, and activation.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store residual connection
        residual = x
        
        # First layer: Linear -> Norm -> Activation
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        # Second layer: Linear -> Norm
        out = self.linear2(out)
        out = self.norm2(out)
        
        # Add residual connection and apply final activation
        out = out + residual
        out = self.activation(out)
        
        return out