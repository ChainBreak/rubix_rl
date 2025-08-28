from src.recorder import Recorder, RecorderConfig
import random
from src import rubiks_cube as rc
import numpy as np
from pydantic import BaseModel
from src import player as player_module
import time

class CollectConfig(BaseModel):
    num_samples: int
    max_scramble_steps: int
    recorder: RecorderConfig

def collect_states(config: CollectConfig, checkpoint: str, device: str):
    player = player_module.Player(checkpoint, device)
    num_samples = config.num_samples

    recorder = Recorder(config.recorder)

    cube_generator = generate_random_cubes(config, player)
    for _ in range(num_samples):
        cube = next(cube_generator)
        recorder.record(cube.get_state_dict())

def generate_random_cubes(config:CollectConfig, player:player_module.Player):
    num_cubes = 256
    cubes: list[rc.RubiksCube] = [get_scramble_cube(config) for _ in range(num_cubes)]
    steps_to_go = player.get_steps_to_go(cubes)

    while True:
        actions, next_steps_to_go = player.get_actions(cubes)
        [cube.take_action(int(action.item())) for cube, action in zip(cubes, actions)]
        yield from cubes
        stuck_cubes = next_steps_to_go >= steps_to_go
        steps_to_go = next_steps_to_go
        for i in range(len(cubes)):
            if stuck_cubes[i]:
                cubes[i] = get_scramble_cube(config)
                steps_to_go[i] = config.max_scramble_steps

def get_scramble_cube(config:CollectConfig) -> rc.RubiksCube:
    cube = rc.RubiksCube()
    num_scramble_steps = random.randint(0, config.max_scramble_steps+1)
    for i in range(num_scramble_steps):
        cube.take_action(random.randint(0, cube.action_space_size - 1))

    return cube

