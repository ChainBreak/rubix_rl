"""
Lightning module for Rubik's Cube RL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import omegaconf
import lightning as L
from typing import Any, Dict, Optional, Tuple, Union, override
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

    def create_model(self, config: omegaconf.DictConfig)->nn.Module:

        return nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:

        
        loss = torch.tensor(0.0, requires_grad=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    
    
    def configure_optimizers(self) ->torch.optim.Optimizer:
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