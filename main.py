#!/usr/bin/env python3
"""
Rubik's Cube Reinforcement Learning Solver - CLI Interface

This is the main CLI interface for the Rubik's cube RL solver project.
It provides commands for training, evaluation, and testing the DQN agent.
"""

import click
import torch
from pathlib import Path
from src import rubiks_cube as rc
from omegaconf import OmegaConf

from src import collect_states
from src import trainer
from src import player as player_module

@click.group()
@click.version_option(version='1.0.0')
def cli():
    pass


@cli.command()
@click.option('--config', help='Path to the config file')
def train(config):
    config = OmegaConf.load(config)
    trainer.train(config)

  
@cli.command()
@click.option('--config', help='Path to the config file')
def collect(config):
    config = OmegaConf.load(config)
    collect_states.collect_states(config.collect)

def run_agent():
    pass

@cli.command()
@click.option('--checkpoint', help='Path to the checkpoint file')
@click.option('--device', help='Device to use for the player')
def play(checkpoint, device):
    cube = rc.RubiksCube()
    player = player_module.Player(checkpoint, device)
    while True:
        print(cube)
        print(" ".join(cube.action_space))
        actions = player.get_actions([cube])
        model_operation = cube.action_space[actions[0]]
        print(f"Enter the operations to perform on the cube ({model_operation}):")
        operations = input()
        print(operations)
        if operations.strip() == "":
            operations = [model_operation]
        else:
            operations = operations.split(" ")
        cube.perform_operations(operations)

if __name__ == "__main__":
    cli()
