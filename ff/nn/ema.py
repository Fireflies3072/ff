import torch
import torch.nn as nn
from copy import deepcopy
from contextlib import contextmanager

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = None):
        """
        Args:
            model: Online model
            decay: EMA decay rate
            device: Device for EMA model (e.g. 'cpu' or 'cuda')
        """
        # Deep copy the model structure
        self.ema_model = deepcopy(model)
        self.decay = decay
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.ema_model.to(self.device)
        
        # No gradients, enter eval mode
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update the weights of model to self.ema_model
        """
        # Get state_dicts of online and ema models
        online_dict = model.state_dict()
        ema_dict = self.ema_model.state_dict()

        for name in online_dict:
            # Get the weights of online and ema models
            online_v = online_dict[name]
            ema_v = ema_dict[name]
            # Move the weights to the same device as the ema model
            online_v_on_ema_device = online_v.to(self.device, non_blocking=True)

            # Update the weights of ema model
            if ema_v.is_floating_point():
                ema_v.lerp_(online_v_on_ema_device, 1.0 - self.decay)
            else:
                ema_v.copy_(online_v_on_ema_device)

    def apply_to(self, model: nn.Module):
        """
        Load the weights of ema model back to the training model (for evaluation or inference)
        Note: This usually involves data transfer from CPU to GPU
        """
        model.load_state_dict(self.ema_model.state_dict())
    
    @contextmanager
    def ema_scope(self, model: nn.Module):
        """
        EMA scope with adaptive device check
        """
        # Get the device of the online model
        model_device = next(model.parameters()).device
        
        # Use EMA model if the device is the same as the online model
        if self.device == model_device:
            yield self.ema_model
        else:
            # If cross-device, then backup-overwrite-restore
            backup_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            try:
                # Load the EMA model to the online model
                model.load_state_dict(self.ema_model.state_dict())
                yield model
            finally:
                # Restore the original state
                model.load_state_dict(backup_state)
                del backup_state
