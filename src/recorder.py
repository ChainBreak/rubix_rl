import json
import atexit
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

class RecorderConfig(BaseModel):
    dataset_dir: str
    items_per_file: int


class Recorder:
    def __init__(self, config: RecorderConfig): 
        self.dataset_dir: Path = Path(config.dataset_dir)
        self.buffer = []
        self.items_per_file: int = config.items_per_file
        
        self.exit_hooks_registered = False

    def record(self, item: dict):
        # Register atexit hook to save buffer on program exit
        if not self.exit_hooks_registered:
            atexit.register(self._save_on_exit)
            self.exit_hooks_registered = True

        self.buffer.append(item)
        
        if len(self.buffer) >= self.items_per_file:
            self.save()
            self.buffer = []

    def save(self):
        if self.buffer:  # Only save if buffer has data
            # Ensure directory exists
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

            recording_path = self.dataset_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            
            with open(recording_path, "w") as f:
                json.dump(self.buffer, f)

            print(f"Saved {len(self.buffer)} items to {recording_path}")
    
    def _save_on_exit(self):
        """Private method called by atexit to save remaining buffer contents"""

        print(f"Program exiting, saving...")
        self.save()