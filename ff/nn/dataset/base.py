import torch
from torch.utils.data import Dataset

class DatasetGeneric(Dataset):
    def __init__(self, dataset, logic_map):
        self.dataset = dataset
        self.logic_map = logic_map

        # Logic registry
        if not hasattr(self, '_logic_registry'):
            self._logic_registry = {}
        self._logic_registry['data_f32'] = self._handle_data_f32
        self._logic_registry['data_i64'] = self._handle_data_i64

        # Create handler map
        self.handler_map = {}
        logic_map = logic_map or {}
        for key, logic in logic_map.items():
            # Assign handler
            if logic is None:
                continue
            elif logic in self._logic_registry:
                self.handler_map[key] = self._logic_registry[logic]
            else:
                self.handler_map[key] = logic

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
    
    def _handle_data_f32(self, data):
        return torch.tensor(data, dtype=torch.float32)
    
    def _handle_data_i64(self, data):
        return torch.tensor(data, dtype=torch.int64)
