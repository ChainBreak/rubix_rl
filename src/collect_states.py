from src.recorder import Recorder
import random
from src import rubiks_cube as rc
import numpy as np

def collect_states(config):
    recorder = Recorder(config.recorder)

    for _ in range(config.num_samples):
        randomly_scramble_cube_and_record_states(recorder, config.max_scramble_steps)   

def randomly_scramble_cube_and_record_states(recorder:Recorder, max_scramble_steps:int):
    cube = rc.RubiksCube()
    for i in range(max_scramble_steps+1):

        save_probability = calculate_save_probability(i, max_scramble_steps)
        
        if random.random() < save_probability:
            recorder.record(cube.get_state_dict())

        cube.take_action(random.randint(0, cube.action_space_size - 1))

def calculate_save_probability(scramble_step: int, max_scramble_steps: int) -> float:

    # Define interpolation points: x=[0, max_steps], y=[0.1, 1.0]
    x_points = [0, max_scramble_steps]
    y_points = [0.1, 1.0]
    
    return float(np.interp(scramble_step, x_points, y_points))
