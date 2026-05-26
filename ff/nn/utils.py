import math
import torch
from pathlib import Path

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def save_model(path, model, optimizer=None, scheduler=None, epoch=None, info=None, simplified_model=None):
    # Create directory
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Generate dictionary
    state = {}

    # Extract model parameters
    if isinstance(model, (list, tuple)):
        state['model'] = [m.state_dict() for m in model]
    else:
        state['model'] = [model.state_dict()]
    # Extract optimizer parameters
    if optimizer is not None:
        if isinstance(optimizer, (list, tuple)):
            state['optimizer'] = [o.state_dict() for o in optimizer]
        else:
            state['optimizer'] = [optimizer.state_dict()]
    # Extract scheduler parameters
    if scheduler is not None:
        if isinstance(scheduler, (list, tuple)):
            state['scheduler'] = [s.state_dict() for s in scheduler]
        else:
            state['scheduler'] = [scheduler.state_dict()]
    # Other information
    if epoch is not None:
        state['epoch'] = epoch
    if info is not None:
        state['info'] = info
    
    # Save model
    torch.save(state, path)
    if simplified_model is not None:
        if isinstance(simplified_model, (list, tuple)):
            torch.save({'model': [m.state_dict() for m in simplified_model]}, p.parent/f'{p.stem}_{epoch}{p.suffix}')
        else:
            torch.save({'model': [simplified_model.state_dict()]}, p.parent/f'{p.stem}_{epoch}{p.suffix}')

def read_model(path, model, optimizer=None, scheduler=None):
    # If path is not a file, return default values
    if not Path(path).is_file():
        return 1, None
    
    # Load checkpoint
    epoch = 1
    checkpoint = torch.load(path)
    # Load model parameters
    if 'model' in checkpoint:
        if isinstance(model, (list, tuple)):
            for i in range(len(model)):
                if model[i] is not None:
                    model[i].load_state_dict(checkpoint['model'][i])
        else:
            model.load_state_dict(checkpoint['model'][0])
    # Load optimizer parameters
    if (optimizer is not None) and ('optimizer' in checkpoint):
        if isinstance(optimizer, (list, tuple)):
            for i in range(len(optimizer)):
                if optimizer[i] is not None:
                    optimizer[i].load_state_dict(checkpoint['optimizer'][i])
        else:
            optimizer.load_state_dict(checkpoint['optimizer'][0])
    # Load scheduler parameters
    if (scheduler is not None) and ('scheduler' in checkpoint):
        if isinstance(scheduler, (list, tuple)):
            for i in range(len(scheduler)):
                if scheduler[i] is not None:
                    scheduler[i].load_state_dict(checkpoint['scheduler'][i])
        else:
            scheduler.load_state_dict(checkpoint['scheduler'][0])
    # Load other parameters
    epoch = 1 if 'epoch' not in checkpoint else checkpoint['epoch'] + 1
    info = None if 'info' not in checkpoint else checkpoint['info']
    
    return epoch, info
