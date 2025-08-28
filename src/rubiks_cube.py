import torch
import rubik_sim
import textwrap
from typing import Any

class RubiksCube:
    
    def __init__(self):
        self.cube: rubik_sim.RubiksCube = rubik_sim.RubiksCube()
        self.action_space = list(self.cube._moves_notation.keys())
        self.action_space_size = len(self.action_space)
        self.steps_taken = 0

    def take_action(self, action: int):
        operation = self.action_space[action]
        self.perform_operations([operation])
        

    def get_state_dict(self):
        return {
            "steps": self.steps_taken,
            "state": self.cube.color_code
        }

    def load_state_dict(self,state_dict: dict[str, Any])->None:
        self.cube = rubik_sim.RubiksCube.from_color_code(state_dict["state"])
        self.steps_taken = state_dict["steps"]
    
    def get_state_as_tensor(self):
        return convert_color_code_to_tensor(self.cube.color_code)


    def get_all_next_states_as_tensor(self)->torch.Tensor:
        next_states_color_codes = self.get_color_codes_of_all_next_states()
        next_state_tensors = [convert_color_code_to_tensor(color_code) for color_code in next_states_color_codes]
        return torch.stack(next_state_tensors)


    def get_color_codes_of_all_next_states(self)->list[str]:
        next_states_color_codes = []
        for action in self.action_space:
            next_cube = rubik_sim.RubiksCube.from_color_code(self.cube.color_code)
            next_cube.perform_operations([action])
            next_states_color_codes.append(next_cube.color_code)
        return next_states_color_codes


    def perform_operations(self, operations: list[str]):
        self.cube.perform_operations(operations)
        self.steps_taken += 1

        if self.cube.is_solved():
            self.steps_taken = 0


    def __str__(self):
        color_code = self.cube.color_code
        return convert_color_code_to_emoji_string(color_code)


def convert_color_code_to_tensor(color_code: str):
    one_hot_tensor_list = [color_char_to_one_hot_map[color_char] for color_char in color_code]
    return torch.cat(one_hot_tensor_list, dim=0)

def convert_color_code_to_emoji_string(color_code: str):
    e = "".join([color_char_to_emoji_map[color_char] for color_char in color_code])
        
    return textwrap.dedent(f"""
        ⬛️⬛️⬛️{e[0:3]}⬛️⬛️⬛️⬛️⬛️⬛️
        ⬛️⬛️⬛️{e[3:6]}⬛️⬛️⬛️⬛️⬛️⬛️
        ⬛️⬛️⬛️{e[6:9]}⬛️⬛️⬛️⬛️⬛️⬛️
        {e[9:12]}{e[18:21]}{e[27:30]}{e[36:39]}
        {e[12:15]}{e[21:24]}{e[30:33]}{e[39:42]}
        {e[15:18]}{e[24:27]}{e[33:36]}{e[42:45]}
        ⬛️⬛️⬛️{e[45:48]}⬛️⬛️⬛️⬛️⬛️⬛️
        ⬛️⬛️⬛️{e[48:51]}⬛️⬛️⬛️⬛️⬛️⬛️
        ⬛️⬛️⬛️{e[51:54]}⬛️⬛️⬛️⬛️⬛️⬛️
        """)

color_char_to_one_hot_map = {
    "G": torch.tensor([1, 0, 0, 0, 0, 0]),
    "O": torch.tensor([0, 1, 0, 0, 0, 0]),
    "Y": torch.tensor([0, 0, 1, 0, 0, 0]),
    "R": torch.tensor([0, 0, 0, 1, 0, 0]),
    "W": torch.tensor([0, 0, 0, 0, 1, 0]),
    "B": torch.tensor([0, 0, 0, 0, 0, 1])
}

color_char_to_emoji_map = {
    "G": "🟩",
    "O": "🟧",
    "Y": "🟨",
    "R": "🟥",
    "W": "⬜",
    "B": "🟦"
}

