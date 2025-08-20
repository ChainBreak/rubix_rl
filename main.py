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

@click.group()
@click.version_option(version='1.0.0')
def cli():
    pass


@cli.command()
@click.option('--config', help='Path to the config file')
def train(config):
    pass
    
@cli.command()
@click.option('--config', help='Path to the config file')
def collect(config):
    config = OmegaConf.load(config)
    collect_states.collect_states(config.collect)

def run_agent():
    pass

@cli.command()
def play():
    cube = rc.RubiksCube()
    while True:
        print(cube)
        print(cube.action_space)
        print("Enter the operations to perform on the cube:")
        operations = input()
        operations = operations.split(" ")
        cube.perform_operations(operations)

if __name__ == "__main__":
    cli()
