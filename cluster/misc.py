from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollator:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, Any]]]) -> Dict[str, torch.Tensor]:
        input_features = [{
            "input_features": feature["input_values"],
        } for feature in features]

        batch = self.processor.pad(input_features, return_tensors="pt")

        labels = [feature["labels"] for feature in features]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
