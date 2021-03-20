import torch

def load_model(ckpt_path, model, optimizer=None, scheduler=None):
        """                                                                                                                                                                
    Function that loads model, optimizer, and scheduler state-dicts from a ckpt.                                                                                       
    Args:                                                                                                                                                              
        model (nn.Module): Initialized model objects                                                                                                                   
        ckpt_path (str): Path to saved model ckpt                                                                                                                      
        optimizer (torch.optim.Optimizer): Initialized optimizer object                                                                                                
        scheduler (torch.optim.lr_scheduler): Initilazied scheduler object                                                                                             
    """
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            optimizer.load_state_dict(checkpoint['scheduler'])
            print(f"Loaded {checkpoint['model_class']} from {ckpt_path} at {checkpoint['ckpt_info']['epoch']}")
