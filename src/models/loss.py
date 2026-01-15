import torch
import torch.nn as nn


class TrafficLoss(nn.Module):
    def __init__(self):
        super(TrafficLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='sum') 

    def forward(self, pred_hm, pred_wh, pred_reg, batch):
        device = pred_hm.device
        gt_hm = batch['hm'].to(device)
        gt_wh = batch['wh'].to(device)
        gt_reg = batch['reg'].to(device)
        mask = batch['reg_mask'].to(device) 
        pos_inds = gt_hm.eq(1).float()  
        neg_inds = gt_hm.lt(1).float() 
        neg_weights = torch.pow(1 - gt_hm, 4)
        pred_hm = torch.clamp(pred_hm, 1e-6, 1 - 1e-6)
        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, 2) * pos_inds
        neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, 2) * neg_weights * neg_inds
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss_hm = -neg_loss
        else:
            loss_hm = -(pos_loss + neg_loss) / num_pos
        mask = mask.unsqueeze(1).float()
        reg_weight = 1.0 / (num_pos + 1e-4)
        loss_off = self.l1_loss(pred_reg * mask, gt_reg * mask) * reg_weight
        loss_wh = self.l1_loss(pred_wh * mask, gt_wh * mask) * reg_weight
        loss = loss_hm + 1.0 * loss_off + 0.1 * loss_wh
        
        return loss, loss_hm, loss_wh, loss_off
