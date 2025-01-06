import torch
import torch.nn as nn
import torch.nn.functional as F


class RVAEHungarianObjective(nn.Module):
    """Object for hungarian loss (ie. lw + bce) for RVAE model."""

    def __init__(self, config):
        """
        Initialize hungarian loss object.
        :param config: config dataclass of RVAE
        """
        self._config = config

    def compute(self, predictions, targets, matchings, tgt_type):

        pred_states = predictions[tgt_type]['vector']  # B, Q, D
        pred_logits = predictions[tgt_type]['mask']    # B, Q
        gt_states = targets[tgt_type]['vector']        # B, num_obj, D 
        gt_mask = targets[tgt_type]['mask']            # B, num_obj

        matching = matchings[f"{tgt_type.lower()}_matching"]

        # Arrange predictions and targets according to matching
        indices, permutation_indices = matching, _get_src_permutation_idx(matching)

        pred_states_idx = pred_states[permutation_indices]
        gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

        pred_logits_idx = pred_logits[permutation_indices]
        gt_mask_idx = torch.cat([t[i] for t, (_, i) in zip(gt_mask, indices)], dim=0).float()

        # calculate CE and L1 Loss
        l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
        if tgt_type == 'LANES':
            l1_loss = l1_loss.sum(-1).mean(-1) * gt_mask_idx
            ce_weight, reconstruction_weight = self._config.line_ce_weight, self._config.line_reconstruction_weight
        else:
            l1_loss = l1_loss.sum(-1) * gt_mask_idx
            ce_weight, reconstruction_weight = self._config.box_ce_weight, self._config.box_reconstruction_weight

        ce_loss = F.binary_cross_entropy_with_logits(pred_logits_idx, gt_mask_idx, reduction="none")

        # Whether to average by batch size or entity count
        bs = gt_mask.shape[0]
        if self._config.norm_by_count:
            num_gt_instances = gt_mask.float().sum(-1)
            num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1
            l1_loss = l1_loss.view(bs, -1).sum() / num_gt_instances
            ce_loss = ce_loss.view(bs, -1).sum() / num_gt_instances
        else:
            l1_loss = l1_loss.view(bs, -1).mean()
            ce_loss = ce_loss.view(bs, -1).mean()

        tgt_type = tgt_type.lower()
        return {f"l1_{tgt_type}": reconstruction_weight * l1_loss, f"ce_{tgt_type}": ce_weight * ce_loss}


def _get_src_permutation_idx(indices):
    """Helper function for permutation of matched indices."""

    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])

    return batch_idx, src_idx



class KLObjective(nn.Module):
    """Kullback-Leibler divergence objective for VAEs."""

    def __init__(self, config):
        """
        Initialize KL objective.
        :param weight: scalar for loss weighting (aka. β)
        :param scenario_type_loss_weighting: scenario-type specific loss weights (ignored).
        """
        self._config = config

    def compute(self, predictions, targets, matchings):

        mu, log_var = predictions["mu"],  predictions["log_var"]
        kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        return {"kl_latent": self._config.kl_weight * kl_loss}


# class RVAEEgoObjective(AbstractCustomObjective):
#     """Simple regression loss of ego attributes (ie. lw + bce)."""

#     def __init__(self, weight: float, scenario_type_loss_weighting: Dict[str, float]):
#         """
#         Initialize ego objective.
#         :param weight: scalar for loss weighting
#         :param scenario_type_loss_weighting: scenario-type specific loss weights (ignored)
#         """
#         self._weight = weight
#         self._scenario_type_loss_weighting = scenario_type_loss_weighting

#     def compute(
#         self, predictions: FeaturesType, targets: TargetsType, matchings: TargetsType, scenarios: ScenarioListType
#     ) -> Dict[str, torch.Tensor]:
#         """Inherited, see superclass."""

#         pred_ego_element: SledgeVectorElement = predictions["sledge_vector"].ego
#         gt_ego_element: SledgeVectorElement = targets["sledge_vector"].ego
#         l1_loss = F.l1_loss(pred_ego_element.states, gt_ego_element.states[..., 0])

#         return {"l1_ego": self._weight * l1_loss}
