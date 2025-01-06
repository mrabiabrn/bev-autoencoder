import logging
from typing import Any, Dict, Union
from hydra.utils import instantiate
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl

import torch.nn as nn

from nuplan.planning.script.builders.lr_scheduler_builder import build_lr_scheduler

from .model import RVAEModel
from .matching import RVAEHungarianMatching
from .objective import RVAEHungarianObjective, KLObjective
from .metric import KLMetric

logger = logging.getLogger(__name__)


class AutoencoderWrapper(nn.Module):
    """
    Custom lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(self, cfg):
        """
        Initialize lightning autoencoder wrapper.
        :param model: autoencoder torch module wrapper.
        :param objectives: list of autoencoder objectives computed at each step
        :param metrics: optional list of metrics to track
        :param matchings: optional list of matching objects (e.g. for hungarian objectives)
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: _description_, defaults to None
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = RVAEModel(cfg)
        self.matcher = RVAEHungarianMatching(cfg)
        self.objectives = [RVAEHungarianObjective(cfg), KLObjective(cfg)]
        self.metric = KLMetric()
       
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warm_up_lr_scheduler = warm_up_lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode

        self.tgt_types = ['LANE', 'VEHICLE']

    def forward(self, batch, prefix="train"):
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets = batch

        predictions = self.model(features)                                                  # ((lane_vector, lane_mask), (vehicle_vector, vehicle_mask))
        matchings = self._compute_matchings(predictions["vector"], targets)                 # {'LANE_matching': [(tensor, tensor)], 'VEHICLE_matching': [(tensor, tensor)]}
        objectives = self._compute_objectives(predictions["vector"], targets, matchings)
        metrics = self._compute_metrics(predictions["latent"], targets, matchings)
        loss = torch.stack(list(objectives.values())).sum()

        # self._log_step(loss, objectives, metrics, prefix)

        logs = {f"total_loss": loss}
        logs.update(objectives)
        logs.update(metrics)

        return logs

    def _compute_objectives(self, predictions, targets, matchings, tgt_type):
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

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        if self.optimizer is None:
            raise RuntimeError("To train, optimizer must not be None.")

        # Get optimizer
        optimizer: Optimizer = instantiate(
            config=self.optimizer,
            params=self.parameters(),
            lr=self.optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )
        # Log the optimizer used
        logger.info(f"Using optimizer: {self.optimizer._target_}")

        # Get lr_scheduler
        lr_scheduler_params: Dict[str, Union[_LRScheduler, str, int]] = build_lr_scheduler(
            optimizer=optimizer,
            lr=self.optimizer.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler,
            lr_scheduler_cfg=self.lr_scheduler,
        )
        lr_scheduler_params["interval"] = "step"
        lr_scheduler_params["frequency"] = 1

        optimizer_dict: Dict[str, Any] = {}
        optimizer_dict["optimizer"] = optimizer
        if lr_scheduler_params:
            logger.info(f"Using lr_schedulers {lr_scheduler_params}")
            optimizer_dict["lr_scheduler"] = lr_scheduler_params

        return optimizer_dict if "lr_scheduler" in optimizer_dict else optimizer_dict["optimizer"]