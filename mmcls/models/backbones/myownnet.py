import torch.nn as nn

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class MyOwnNet(BaseBackbone):
    """
    A toy model for CIFAR.
    """

    def __init__(self):
        super(MyOwnNet,self).__init__()
        #self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        )
        '''
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(64*32*32,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024,num_classes),
            )
            '''
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16*8*8)
        
        '''
        if self.num_classes > 0:
            x = x.view(x.size(0), 64*32*32)
            x = self.classifier(x)
            '''
        return x