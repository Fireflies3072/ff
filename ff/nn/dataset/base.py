import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetGeneric(Dataset):
    def __init__(self, dataset, logic_map):
        self.dataset = dataset
        self.logic_map = logic_map

        # Logic registry
        if not hasattr(self, '_logic_registry'):
            self._logic_registry = {}
        self._logic_registry['to_tensor'] = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x)
        self._logic_registry['data_f32'] = lambda x: torch.tensor(x, dtype=torch.float32)
        self._logic_registry['data_i64'] = lambda x: torch.tensor(x, dtype=torch.int64)
        self._logic_registry['to_f32'] = lambda x: x.to(dtype=torch.float32)
        self._logic_registry['to_i64'] = lambda x: x.to(dtype=torch.int64)

        # Create handler map
        self.handler_map = {}
        logic_map = logic_map or {}
        for key, logic in logic_map.items():
            # Assign handler
            if logic is None:
                continue
            logics = logic if isinstance(logic, list) else [logic]
            self.handler_map[key] = self._build_pipeline(logics)

    def __getitem__(self, index):
        data = dict(self.dataset[index])
        return self._apply_handler(data)

    def __len__(self):
        return len(self.dataset)
    
    def _apply_handler(self, data):
        for key, handler in self.handler_map.items():
            if handler is not None:
                data[key] = handler(data[key])
        return data
    
    def _build_pipeline(self, logics):
        handlers = [self._logic_registry[l] if isinstance(l, str) else l for l in logics]
        def pipeline(data):
            for h in handlers:
                data = h(data)
            return data
        return pipeline
