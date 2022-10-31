from abc import ABC, abstractclassmethod
import torch

class BaseModelDNN(ABC):
    @abstractclassmethod
    def __init__(self) -> None:
        pass
    
    def eval_mode(self):
        self.net.eval()
    def train_mode(self):
        self.net.train()
        
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
    def load_networks(self, path):
        self.checkpoint = torch.load(path)
        if len(self.GPU_IDs) == 1:
            self.net.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.net.module.load_state_dict(self.checkpoint['state_dict'])
            
    def resume_training(self, path, net_only=True):
        self.load_networks(path)
        if not net_only:
            self.start_epoch = self.checkpoint['stop_epoch']
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.last_epoch = self.start_epoch

