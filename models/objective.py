import torch
import torch.nn as nn
import torch.nn.functional as F


class RVAEHungarianObjective(nn.Module):
    """Object for hungarian loss (ie. lw + bce) for RVAE model."""

    def __init__(self, config):
        super().__init__()
        """
        Initialize hungarian loss object.
        :param config: config dataclass of RVAE
        """
        self._config = config

    def forward(self, predictions, targets, matchings, tgt_type):

        pred_states = predictions[tgt_type]['vector']  # B, Q, D
        pred_logits = predictions[tgt_type]['mask']    # B, Q
        gt_states = targets[tgt_type]['vector']        # B, MAX, D 
        gt_mask = targets[tgt_type]['mask']            # B, MAX

        matching = matchings[f"{tgt_type.lower()}_matching"]

        # Arrange predictions and targets according to matching
        indices, permutation_indices = matching, _get_src_permutation_idx(matching)

        pred_states_idx = pred_states[permutation_indices]                                    # N, D
        gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)   

        pred_logits_idx = pred_logits[permutation_indices]                                    # N       
        gt_mask_idx = torch.cat([t[i] for t, (_, i) in zip(gt_mask, indices)], dim=0).float()   

        # calculate CE and L1 Loss
        if tgt_type == 'LANES':
            l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")       # B * num_queries, D
            l1_loss = l1_loss.sum(-1).mean(-1) * gt_mask_idx
            ce_weight, reconstruction_weight = self._config.line_ce_weight, self._config.line_reconstruction_weight
        else:
            l1_loss = F.l1_loss(pred_states_idx[:,[0,1,3,4]], gt_states_idx[:,[0,1,3,4]], reduction="none")       # B * num_queries, D
            l1_loss = l1_loss * torch.tensor([1, 1, 0.25, 0.25]).to(l1_loss.device)
            l1_loss = l1_loss.sum(-1) * gt_mask_idx             
            ce_weight, reconstruction_weight = self._config.box_ce_weight, self._config.box_reconstruction_weight

            pred_angle_in_rad = pred_states_idx[:,2] * torch.pi    # -pi  to pi     B * num_queries
            target_angle_in_rad = gt_states_idx[:,2] * torch.pi    # -pi to pi      B * num_queries
            diff_angle = pred_angle_in_rad - target_angle_in_rad   # -2pi to 2pi
            angle_error = - torch.pi + torch.fmod(diff_angle + torch.pi, 2 * torch.pi)  
            loss_angle = F.l1_loss(angle_error, torch.zeros_like(angle_error), reduction='none')     # B * num_queries
            loss_angle = loss_angle * gt_mask_idx
            angle_weight = self._config.box_angle_weight

            # loss_giou = 1 - torch.diag(generalized_box_iou(
            #                                                 box_cxcywh_to_xyxy(pred_states_idx[:,[0,1,3,4]]),
            #                                                 box_cxcywh_to_xyxy(gt_states_idx[:,[0,1,3,4]])
            #                                                 ))     
            # loss_giou = loss_giou.sum(-1) * gt_mask_idx     # 
            

        ce_loss = F.binary_cross_entropy_with_logits(pred_logits_idx, gt_mask_idx, reduction="none")

        # Whether to average by batch size or entity count
        bs = gt_mask.shape[0]
        if self._config.norm_by_count:
            num_gt_instances = gt_mask.float().sum()   # num of instances in batch
            num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1
            l1_loss = l1_loss.view(bs, -1).sum() / num_gt_instances
            ce_loss = ce_loss.view(bs, -1).sum() / num_gt_instances
            if tgt_type == 'VEHICLES':
                loss_angle = loss_angle.view(bs, -1).sum() / num_gt_instances
        else:
            l1_loss = l1_loss.view(bs, -1).mean()
            ce_loss = ce_loss.view(bs, -1).mean()
            if tgt_type == 'VEHICLES':
                loss_angle = loss_angle.view(bs, -1).mean()
        

        tgt_type = tgt_type.lower()
        out = {
                f"l1_{tgt_type}": reconstruction_weight * l1_loss, 
                f"ce_{tgt_type}": ce_weight * ce_loss
                }
        if tgt_type == 'vehicles':
            out[f"angle_{tgt_type}"] = angle_weight * loss_angle
        
        return out

    def compute(self, predictions, targets, matchings, tgt_type):

        out = self.forward(
                           predictions, 
                           targets, 
                           matchings, 
                           tgt_type
                           )

        return out
    


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


from torchvision.ops.boxes import box_area

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)     

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area



def _get_src_permutation_idx(indices):
    """Helper function for permutation of matched indices."""

    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])

    return batch_idx, src_idx



class KLObjective(nn.Module):
    """Kullback-Leibler divergence objective for VAEs."""

    def __init__(self, config):
        super().__init__()
        """
        Initialize KL objective.
        :param weight: scalar for loss weighting (aka. β)
        :param scenario_type_loss_weighting: scenario-type specific loss weights (ignored).
        """
        self._config = config
    
    def forward(self, predictions, targets, matchings):
        pass

    def compute(self, predictions, targets, matchings):

        mu, log_var = predictions["mu"],  predictions["log_var"]
        kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        return {"kl_latent": self._config.kl_weight * kl_loss}
