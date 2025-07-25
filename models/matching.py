import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class RVAEHungarianMatching(nn.Module):
    """Object for hungarian matching of RVAE model"""

    def __init__(self, config):
        super().__init__()
        """
        Initialize matching object of RVAE
        :param key: string identifier if sledge vector dataclass
        :param config: config dataclass of RVAE
        """
        self._config = config

    def forward(self, predictions, targets):
        pass

    @torch.no_grad()
    def compute(self, predictions, targets, tgt_type):

        pred_states = predictions[tgt_type]['vector']  # B, Q, D
        gt_states = targets[tgt_type]['vector']        # B, num_obj, D 
        pred_logits = predictions[tgt_type]['mask']    # B, Q
        if tgt_type == 'VEHICLES' and self._config.multiclass:
            gt_mask = targets[tgt_type]['class']           # B, num_obj, C
        else:
            gt_mask = targets[tgt_type]['mask']            # B, num_obj
        

        if tgt_type in ['LANES', 'LANE_DIVIDERS']:
            l1_cost = _get_line_l1_cost(gt_states, pred_states, gt_mask)
            ce_weight, reconstruction_weight = self._config.line_ce_weight, self._config.line_reconstruction_weight
            ce_cost = _get_ce_cost(gt_mask, pred_logits)
        else:
            l1_cost = _get_box_l1_cost(gt_states, pred_states, gt_mask)
            ce_weight, reconstruction_weight = self._config.box_ce_weight, self._config.box_reconstruction_weight
            # gt_classes = targets[tgt_type]['class']
            # ce_cost = _get_ce_cost_multiclass(gt_classes, pred_logits)

            if self._config.multiclass:
                ce_cost = _get_ce_cost_multiclass(gt_mask, pred_logits)
            else:
                ce_cost = _get_ce_cost(gt_mask, pred_logits)
        
        cost = ce_weight * ce_cost + reconstruction_weight * l1_cost
        cost = cost.cpu()  # NOTE: This unfortunately is the runtime bottleneck

        indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
        matching = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        tgt_type = tgt_type.lower()
        return {f"{tgt_type}_matching": matching}


@torch.no_grad()
def _get_ce_cost(gt_mask: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Calculated cross-entropy matching cost based on numerically stable PyTorch version, see:
    https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    :param gt_mask: ground-truth binary existence labels, shape: (batch, num_gt)
    :param pred_logits: predicted (normalized) logits of existence, shape: (batch, num_pred)
    :return: cross-entropy cost tensor of shape (batch, num_pred, num_gt)
    """

    gt_mask_expanded = gt_mask[:, :, None].detach().float()  # (b, ng, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, np)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val))
    ce_cost = (1 - gt_mask_expanded) * pred_logits_expanded + helper_term  # (b, ng, np)
    ce_cost = ce_cost.permute(0, 2, 1)  # (b, np, ng)

    return ce_cost

@torch.no_grad()
def _get_ce_cost_multiclass(gt_labels: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy matching cost for multi-class classification.

    :param gt_labels: Ground-truth class indices (not one-hot), shape: (batch, num_gt)
    :param pred_logits: Predicted logits (not softmaxed), shape: (batch, num_pred, num_classes)
    :return: cross-entropy cost tensor of shape (batch, num_pred, num_gt)
    """

    b, np, nc = pred_logits.shape  # batch, num_pred, num_classes
    _, ng = gt_labels.shape  # batch, num_gt

    gt_labels_expanded = gt_labels[:, None, :].long().expand(b, np, ng)  # Shape: (b, ng, 1)
    log_probs = pred_logits.log_softmax(dim=-1)  # (b, np, num_classes)
    ce_cost = -log_probs[torch.arange(b)[:, None, None], torch.arange(np)[None, :, None], gt_labels_expanded]  # (b, np, ng)

    return ce_cost



@torch.no_grad()
def _get_line_l1_cost(gt_states: torch.Tensor, pred_states: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the L1 matching cost for line state tensors.
    :param gt_states: ground-truth line tensor, shape: (batch, num_gt, state_size)
    :param pred_states: predicted line tensor, shape: (batch, num_pred, state_size)
    :param gt_mask: ground-truth binary existence labels for masking, shape: (batch, num_gt)
    :return: L1 cost tensor of shape (batch, num_pred, num_gt)
    """

    gt_states_expanded = gt_states[:, :, None].detach()  # (b, ng, 1, *s)
    pred_states_expanded = pred_states[:, None].detach()  # (b, 1, np, *s)
    l1_cost = gt_mask[..., None] * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1).mean(dim=-1)
    l1_cost = l1_cost.permute(0, 2, 1)  # (b, np, ng)

    return l1_cost


# TODO: Replace box cost with your own BEV-DETR matching cost

@torch.no_grad()
def _get_box_l1_cost(gt_states: torch.Tensor, pred_states: torch.Tensor, gt_mask: torch.Tensor):
    """
    Calculates the L1 matching cost for bounding box state tensors, based on the (x,y) position.
    :param gt_states: ground-truth box tensor, shape: (batch, num_gt, state_size)
    :param pred_states: predicted box tensor, shape: (batch, num_pred, state_size)
    :param gt_mask: ground-truth binary existence labels for masking, shape: (batch, num_gt)
    :param object_indexing: index enum of object type.
    :return: L1 cost tensor of shape (batch, num_pred, num_gt)
    """

    # NOTE: Bounding Box L1 matching only considers position, ignoring irrelevant attr. (e.g. box extent)
    gt_states_expanded = gt_states[:, :, None, [0, 1]].detach()      # (b, ng, 1, 2)
    pred_states_expanded = pred_states[:, None, :, [0, 1]].detach()  # (b, 1, np, 2)
    l1_cost = gt_mask[..., None] * (gt_states_expanded - pred_states_expanded).abs().sum(dim=-1)  # (b, ng, np)
    l1_cost = l1_cost.permute(0, 2, 1)  # (b, np, ng)

    return l1_cost