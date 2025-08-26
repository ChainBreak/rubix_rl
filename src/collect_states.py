from src.recorder import Recorder, RecorderConfig
import random
from src import rubiks_cube as rc
import numpy as np
from pydantic import BaseModel

class CollectConfig(BaseModel):
    num_samples: int
    max_scramble_steps: int
    recorder: RecorderConfig

def collect_states(config: CollectConfig):
    num_samples = config.num_samples
    record_number_of_cube_states(config, num_samples)

def record_number_of_cube_states(config:CollectConfig, num_samples:int):
    recorder = Recorder(config.recorder)
    cube_generator = generate_random_cubes(config)
    for _ in range(num_samples):
        cube = next(cube_generator)
        recorder.record(cube.get_state_dict())

def generate_random_cubes(config:CollectConfig):
    while True:
        yield from scramble_cube(config)


def scramble_cube(config:CollectConfig):
    cube = rc.RubiksCube()
    for i in range(config.max_scramble_steps+1):

        save_probability = calculate_save_probability(i, config.max_scramble_steps)
        
        if random.random() < save_probability:
            yield cube

        cube.take_action(random.randint(0, cube.action_space_size - 1))

def calculate_save_probability(scramble_step: int, max_scramble_steps: int) -> float:

    if scramble_step == 0:
        return 0.1
    else:
        return 1.0
