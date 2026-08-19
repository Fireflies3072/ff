import json
import os
import numpy as np
import torch
from typing import Any

class ExperimentLogger:
    def __init__(self, path: str, auto_save: bool = True):
        """Experiment logger for saving heterogeneous parameters and metric runs to JSON.

        The output file is a JSON array of records. Each record groups one unique
        parameter set with all metric runs logged under those parameters::

            [
                {
                    "params": { ... },   # hyperparameters / config for this group
                    "runs": [            # one dict per log() call with these params
                        { ... },
                        { ... }
                    ]
                },
                ...
            ]

        Example after two log calls with the same params and one with different params::

            [
                {
                    "params": {"lr": 0.01, "model": "resnet"},
                    "runs": [
                        {"psnr": 28.4, "ssim": 0.91},
                        {"psnr": 28.1, "ssim": 0.90}
                    ]
                },
                {
                    "params": {"lr": 0.001, "model": "resnet"},
                    "runs": [
                        {"psnr": 30.2, "ssim": 0.94}
                    ]
                }
            ]

        Args:
            path: Path to the output JSON file.
            auto_save: Whether to automatically flush data to disk on each log call.
        """

        self.path = path
        self.auto_save = auto_save
        self.records: list[dict[str, Any]] = []

        dir = os.path.dirname(os.path.abspath(self.path))
        if dir:
            os.makedirs(dir, exist_ok=True)

        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception as e:
                print(f"[ExperimentLogger] Failed to load existing file, starting fresh: {e}")

    def log(self, params: dict[str, Any], data: dict[str, Any]):
        """Log a data run for a given set of parameters.

        If the parameter set already exists, append the new run data to its run list.
        Otherwise, create a new record entry.

        Args:
            params: Dictionary of parameters/hyperparameters.
            data: Dictionary of metrics or outputs for this specific run.
        """

        # Clean the parameters and data to be serializable
        params = self._convert_to_serializable(params)
        data = self._convert_to_serializable(data)

        # Check if the parameters already exist
        matched_record: dict[str, Any] | None = None
        for record in self.records:
            if record.get("params") == params:
                matched_record = record
                break

        if matched_record is not None:
            matched_record["runs"].append(data)
        else:
            self.records.append({
                "params": params,
                "runs": [data]
            })

        if self.auto_save:
            self.save()

    def get_runs(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Retrieve all recorded runs for a given parameter set."""
        params = self._convert_to_serializable(params)
        for record in self.records:
            if record.get("params") == params:
                return record.get("runs", [])
        return []

    def save(self):
        """Flush in-memory records to the JSON file."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=4, ensure_ascii=False)

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Recursively convert Tensors, Numpy arrays, and custom numbers to native Python types."""
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(v) for v in obj]
        return obj
