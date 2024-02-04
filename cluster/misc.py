from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollator:
    """
    DataCollator is a class that collates data for model training.

    Args:
        processor (Any): The processor used for data preprocessing.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the collated batch of input features and labels.
    """
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, Any]]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of input features and return a dictionary containing the processed batch.

        Args:
            features (List[Dict[str, Union[torch.Tensor, Any]]]): A list of dictionaries containing input features.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the processed batch.

        """
        input_features = [{
            "input_features": feature["input_values"],
        } for feature in features]

        batch = self.processor.pad(input_features, return_tensors="pt")

        labels = [feature["labels"] for feature in features]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
