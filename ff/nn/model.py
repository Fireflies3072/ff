import torch
from pathlib import Path

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def save_model(path, model, optimizer=None, epoch=None, info=None, save_simplied_model=True):
    # Create directory
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Generate dictionary
    state = {}

    # Extract model parameters
    if type(model) == list:
        state['model'] = [m.state_dict() for m in model]
    else:
        state['model'] = [model.state_dict()]
    # Extract optimizer parameters
    if optimizer is not None:
        if type(optimizer) == list:
            state['optimizer'] = [o.state_dict() for o in optimizer]
        else:
            state['optimizer'] = [optimizer.state_dict()]
    # Other information
    if epoch is not None:
        state['epoch'] = epoch
    if info is not None:
        state['info'] = info
    
    # Save model
    torch.save(state, path)
    if save_simplied_model:
        torch.save({'model': state['model']}, p.parent/f'{p.stem}_{epoch}{p.suffix}')

def read_model(path, model, optimizer=None):
    # If path is not a file, return default values
    if not Path(path).is_file():
        return 1, None
    
    # Load checkpoint
    epoch = 1
    checkpoint = torch.load(path)
    # Load model parameters
    if 'model' in checkpoint:
        if type(model) == list:
            for i in range(len(checkpoint['model'])):
                model[i].load_state_dict(checkpoint['model'][i])
        else:
            model.load_state_dict(checkpoint['model'][0])
    # Load optimizer parameters
    if (optimizer is not None) and ('optimizer' in checkpoint):
        if type(optimizer) == list:
            for i in range(len(checkpoint['optimizer'])):
                optimizer[i].load_state_dict(checkpoint['optimizer'][i])
        else:
            optimizer.load_state_dict(checkpoint['optimizer'][0])
    # Load other parameters
    epoch = 1 if 'epoch' not in checkpoint else checkpoint['epoch'] + 1
    info = None if 'info' not in checkpoint else checkpoint['info']
    
    return epoch, info
