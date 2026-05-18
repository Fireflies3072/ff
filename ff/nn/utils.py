import math
import torch
from pathlib import Path

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def save_model(path, model, optimizer=None, epoch=None, info=None, simplified_model=None):
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

def read_model(path, model, optimizer=None):
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
    # Load other parameters
    epoch = 1 if 'epoch' not in checkpoint else checkpoint['epoch'] + 1
    info = None if 'info' not in checkpoint else checkpoint['info']
    
    return epoch, info

def get_sinusoidal_embedding(time_length, embedding_dim):
    """
    Sinusoidal positional encoding
    """
    t = torch.arange(time_length)
    half_dim = embedding_dim // 2
    embedding = math.log(10000) / (half_dim - 1)
    embedding = torch.exp(torch.arange(half_dim) * -embedding)
    embedding = t[:, None] * embedding[None, :]
    embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
    return embedding
