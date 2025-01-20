import logging
from typing import Any, Dict, Union
import torch
import pytorch_lightning as pl

import torch.nn as nn


from .model import RVAEModel
from .matching import RVAEHungarianMatching
from .objective import RVAEHungarianObjective, KLObjective
from .metric import KLMetric

logger = logging.getLogger(__name__)


class RVAEWrapper(nn.Module):

    def __init__(self, cfg):
        """
        Initialize autoencoder wrapper.
        """
        super().__init__()

        self.model = RVAEModel(cfg)
        self.matcher = RVAEHungarianMatching(cfg)
        self.objectives = [RVAEHungarianObjective(cfg), KLObjective(cfg)]
        self.metric = KLMetric()

        self.tgt_types = cfg.target_types # ['LANES', 'VEHICLES']

    def forward(self, batch, prefix="train"):
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets = batch['features'], batch['targets']

        predictions = self.model(features)                                                  # ((lane_vector, lane_mask), (vehicle_vector, vehicle_mask))
        matchings = self._compute_matchings(predictions["vector"], targets)                 # {'LANE_matching': [(tensor, tensor)], 'VEHICLE_matching': [(tensor, tensor)]}
        objectives = self._compute_objectives(predictions, targets, matchings)
        #metrics = self._compute_metrics(predictions["latent"], targets, matchings)
        loss = torch.stack(list(objectives.values())).sum()

        logs = {f"total_loss": loss, "loss_details": {}}
        logs["loss_details"].update(objectives)
        #logs["loss_details"].update(metrics)

        logs.update(matchings)
        logs.update(predictions)

        return logs

    def _compute_objectives(self, predictions, targets, matchings):
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: dictionary of predicted dataclasses.
        :param targets: dictionary of target dataclasses.
        :param matchings: dictionary of prediction-target matching.
        :param scenarios: list of scenario types (for adaptive weighting)
        :return: dictionary of objective names and values
        """
        objectives_dict = {}

        for objective in self.objectives:

            if isinstance(objective, KLObjective):
                objectives_dict.update(objective.compute(predictions["latent"], targets, matchings))
                continue

            for tgt_type in self.tgt_types:
                objectives_dict.update(objective.compute(predictions["vector"], targets, matchings, tgt_type))

        return objectives_dict
    
    def _compute_matchings(self, predictions, targets):
        """
        Computes a the matchings (e.g. for hungarian loss) between prediction and targets.

        :param predictions: dictionary of predicted dataclasses.
        :param targets: dictionary of target dataclasses.
        :return: dictionary of matching names and matching dataclasses
        """
        matchings_dict = {}
        for tgt_type in self.tgt_types:
            matchings_dict.update(self.matcher.compute(predictions, targets, tgt_type))
        return matchings_dict

    def _compute_metrics(self, predictions, targets, matchings):
        """
        Computes a set of metrics used for logging.

        :param predictions: dictionary of predicted dataclasses.
        :param targets: dictionary of target dataclasses.
        :param matchings: dictionary of prediction-target matching.
        :param scenarios: list of scenario types (for adaptive weighting)
        :return: dictionary of metrics names and values
        """
        metrics_dict = {}
        metrics_dict.update(self.metric.compute(predictions, targets, matchings))
        return metrics_dict