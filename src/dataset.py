import json
from pathlib import Path
from typing import Any, override
from torch.utils.data import Dataset
from omegaconf import DictConfig
from src import rubiks_cube as rc

class RubikDataset(Dataset[dict[str, Any]]):
    
    def __init__(self, config: DictConfig) -> None:
      self.cube_states: list[dict[str, Any]] = self.load_all_json_files(config.dataset_dir)

    def load_all_json_files(self, dataset_dir: Path)->list[dict[str, Any]]:
        cube_states = []
        json_files = list(dataset_dir.glob("*.json"))
        for json_file in json_files:
            cube_states.extend(self.load_json_file(json_file))
        return cube_states
    
    def load_json_file(self, json_path: Path)->list[dict[str, Any]]:
      with open(json_path, "r") as f:
        list_of_cube_states =  json.load(f)
        return list_of_cube_states

    
    @override
    def __len__(self)->int:
        return len(self.cube_states)
    
    @override
    def __getitem__(self, idx: int)->dict[str, Any]:

        cube_state = self.cube_states[idx]
        
        cube = rc.RubiksCube()
        cube.load_state_dict(cube_state)

        current_state = cube.get_state_as_tensor()
        next_states = cube.get_all_next_states_as_tensor()

        return {
            "current_state": current_state,
            "next_states": next_states
        }