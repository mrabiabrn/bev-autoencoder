import torch

class KLMetric:
    def __init__(self):
        pass

    def compute(self, predictions, targets, matchings):

        mu, log_var = predictions["mu"], predictions["log_var"]
        kl_metric = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())

        return {"kl_metric": kl_metric}