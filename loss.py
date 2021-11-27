import torch.nn.functional as F
import torch.nn as nn
import math
import torch

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
    def __name__(self):
      return "BCE"

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,ALPHA=0.5,BEETA=0.5):
        super(TverskyLoss, self).__init__()
        self.ALPHA = ALPHA
        self.BEETA = BEETA
    def __name__(self):
      return "TVSKY"
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + self.ALPHA*FP + self.BEETA*FN + smooth)  
        
        return 1 - Tversky


class LogCosDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LogCosDiceLoss, self).__init__()
    def __name__(self):
      return "LCDL"

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        # Dice_BCE = BCE + dice_loss
        return torch.log(torch.cosh(dice_loss))

class CustomLoss1(nn.Module):
    def __init__(self,tverskey_factor,log_cos_dice_factor,binary_cross_entorphy_factor,weight=None, size_average=True):
        super(CustomLoss1, self).__init__()
        self.tverskey_factor = tverskey_factor
        self.log_cos_dice_factor = log_cos_dice_factor
        self.binary_cross_entorphy_factor = binary_cross_entorphy_factor
    def __name__(self):
      return "CUSTOM1"

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs_ = inputs
        targets_ = targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = LogCosDiceLoss().forward(inputs_,targets_)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        tverskey_loss = TverskyLoss(0.5,0.5).forward(inputs_,targets_)
        final_loss = dice_loss*self.log_cos_dice_factor+bce_loss*self.binary_cross_entorphy_factor+tverskey_loss*self.tverskey_factor
        return final_loss


class CustomLoss2(nn.Module):
    def __init__(self,lambda_,weight=None, size_average=True):
        super(CustomLoss2, self).__init__()
        self.lambda_ = lambda_
    def __name__(self):
      return "CUSTOM2"

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        final_loss = dice_loss*self.lambda_+bce_loss*(1-self.lambda_)
        return final_loss