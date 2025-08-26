from src.rubiks_cube import RubiksCube
from src.lit_module import LitModule
import torch
import torch.nn.functional as F

class Player:
    def __init__(self, checkpoint: str, device: str):
        self.model = LitModule.load_from_checkpoint(checkpoint)
        self.device = torch.device(device)
        self.model.to(self.device)

    def get_steps_to_go(self, cubes: list[RubiksCube]) -> list[int]:
        list_of_states = [cube.get_state_as_tensor() for cube in cubes]

        states = torch.stack(list_of_states).float().to(self.device)

        steps_to_go_logits = self.model(states)

        steps_to_go = torch.argmax(steps_to_go_logits, dim=1)

        return steps_to_go.tolist()

    def get_actions(self, cubes: list[RubiksCube]) -> list[str]:
        
        list_of_next_states = [cube.get_all_next_states_as_tensor() for cube in cubes]

        next_states = torch.cat(list_of_next_states, dim=0).float().to(self.device)

        steps_to_go_logits = self.model(next_states)

        steps_to_go_probs = F.softmax(steps_to_go_logits, dim=1)

        steps_to_go = torch.multinomial(steps_to_go_probs, num_samples=1).squeeze(-1)

        # steps_to_go = torch.argmax(steps_to_go_logits, dim=1)
        
        steps_to_go = steps_to_go.reshape(len(cubes), -1)

        actions = steps_to_go.argmin(dim=1)

        return actions.tolist()